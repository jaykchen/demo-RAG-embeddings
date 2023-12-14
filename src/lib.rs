use flowsnet_platform_sdk::logger;
use openai_flows::{
    chat::{ChatModel, ChatOptions},
    embeddings::EmbeddingsInput,
    OpenAIFlows,
};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::str;
use vector_store_flows::*;
use webhook_flows::{create_endpoint, request_handler, send_response};

#[no_mangle]
#[tokio::main(flavor = "current_thread")]
pub async fn on_deploy() {
    create_endpoint().await;
}

#[request_handler]
async fn handler(_headers: Vec<(String, String)>, qry: HashMap<String, Value>, body: Vec<u8>) {
    logger::init();
    let collection_name = qry
        .get("collection_name")
        .unwrap()
        .as_str()
        .unwrap_or("my-book");

    let vector_size = qry.get("vector_size").unwrap().as_str().unwrap();
    let vector_size: u64 = vector_size.parse().unwrap_or(1536);
    let mut id: u64 = 0;

    if qry.contains_key("reset") {
        log::debug!("Reset the database");
        // Delete collection, ignore any error
        _ = delete_collection(collection_name).await;
        // Create collection
        let p = CollectionCreateParams {
            vector_size: vector_size,
        };
        if let Err(e) = create_collection(collection_name, &p).await {
            log::error!(
                "Cannot create collection named: {} with error: {}",
                collection_name,
                e
            );
            send_success("Cannot create collection");
            return;
        }
    } else {
        log::debug!("Continue with existing database");
        match collection_info(collection_name).await {
            Ok(ci) => {
                id = ci.points_count;
            }
            Err(e) => {
                log::error!("Cannot get collection stat {}", e);
                send_success("Cannot query database!");
                return;
            }
        }
    }
    log::debug!("Starting ID is {}", id);

    let mut openai = OpenAIFlows::new();
    openai.set_retry_times(3);
    // let json_contents = include_str!("../segmented_text.json");

    let data: Vec<Vec<String>> = serde_json::from_slice(&body).expect("failed to parse json");

    let mut points = Vec::<Point>::new();
    for block in data {
        for current_section in block {
            let input = EmbeddingsInput::String(current_section.clone());
            match openai.create_embeddings(input).await {
                Ok(r) => {
                    for v in r.iter() {
                        let p = Point {
                            id: PointId::Num(id),
                            vector: v.iter().map(|n| *n as f32).collect(),
                            payload: json!({"text": current_section})
                                .as_object()
                                .map(|m| m.to_owned()),
                        };
                        points.push(p);
                        log::debug!("Created vector {} with length {}", id, v.len());
                        id += 1;
                    }
                }
                Err(e) => {
                    log::error!("OpenAI returned an error: {}", e);
                }
            }
        }
    }
    let points_count = points.len();

    if let Err(e) = upsert_points(collection_name, points).await {
        log::error!("Cannot upsert into database! {}", e);
        send_success("Cannot upsert into database!");
        return;
    }

    match collection_info(collection_name).await {
        Ok(ci) => {
            log::debug!(
                "There are {} vectors in collection `{}`",
                ci.points_count,
                collection_name
            );
            send_success(&format!(
                "Successfully inserted {} records. The collection now has {} records in total.",
                points_count, ci.points_count
            ));
        }
        Err(e) => {
            log::error!("Cannot get collection stat {}", e);
            send_success("Cannot upsert into database!");
        }
    }
    if qry.contains_key("ask") {
        if let Ok(question) = String::from_utf8(body) {
            if let Some(res) = get_answer(&question, &collection_name).await {
                send_success(&res);
            }
        } 
    }
}

fn send_success(body: &str) {
    send_response(
        200,
        vec![(String::from("content-type"), String::from("text/html"))],
        body.as_bytes().to_vec(),
    );
}

async fn get_answer(question: &str, collection_name: &str) -> Option<String> {
    let mut openai = OpenAIFlows::new();
    openai.set_retry_times(3);
    let question_vector = match openai
        .create_embeddings(EmbeddingsInput::String(question.to_string()))
        .await
    {
        Ok(r) => {
            if r.len() < 1 {
                log::error!("OpenAI returned no embedding for the question");
            }
            r[0].iter().map(|n| *n as f32).collect()
        }
        Err(e) => {
            log::error!("OpenAI returned an error: {}", e);
            panic!()
        }
    };

    // Search for embeddings from the question
    let p = PointsSearchParams {
        vector: question_vector,
        limit: 5,
    };
    let mut system_prompt_updated = String::from(
        "You're an AI assistant specialized in answering questions about a preloaded book, here are the most relevant fragments pertaining to the user's question: ",
    );
    match search_points(collection_name, &p).await {
        Ok(sp) => {
            for p in sp.iter() {
                log::debug!(
                    "Received vector score={} and text={}",
                    p.score,
                    first_x_chars(
                        p.payload
                            .as_ref()
                            .unwrap()
                            .get("text")
                            .unwrap()
                            .as_str()
                            .unwrap(),
                        256
                    )
                );
                let p_text = p
                    .payload
                    .as_ref()
                    .unwrap()
                    .get("text")
                    .unwrap()
                    .as_str()
                    .unwrap();
                if p.score > 0.75 && !system_prompt_updated.contains(p_text) {
                    system_prompt_updated.push_str("\n");
                    system_prompt_updated.push_str(p_text);
                }
            }
        }
        Err(e) => {
            log::error!("Vector search returns error: {}", e);
        }
    }
    // log::debug!("The prompt is {} chars starting with {}", system_prompt_updated.len(), first_x_chars(&system_prompt_updated, 256));

    let co = ChatOptions {
        // model: ChatModel::GPT4,
        model: ChatModel::GPT35Turbo16K,
        restart: false,
        system_prompt: Some(&system_prompt_updated),
        post_prompt: None,
        ..Default::default()
    };

    match openai.chat_completion("chatid-99", &question, &co).await {
        Ok(r) => Some(r.choice),

        Err(e) => {
            log::error!("OpenAI returns error: {}", e);
            None
        }
    }
}

fn reply(s: &str) {
    send_response(
        200,
        vec![(String::from("content-type"), String::from("text/html"))],
        s.as_bytes().to_vec(),
    );
}

fn first_x_chars(s: &str, x: usize) -> String {
    s.chars().take(x).collect()
}
