use webhook_flows::{ create_endpoint, request_handler, send_response };
use vector_store_flows::*;
use openai_flows::{ embeddings::{ EmbeddingsInput }, OpenAIFlows, chat };
use serde_json::{ json, Value };
use std::collections::HashMap;
use std::str;
use flowsnet_platform_sdk::logger;

static CHAR_SOFT_LIMIT: usize = 20000;
static CHAR_SOFT_MINIMUM: usize = 100;

#[no_mangle]
#[tokio::main(flavor = "current_thread")]
pub async fn on_deploy() {
    create_endpoint().await;
}

#[request_handler]
async fn handler(_headers: Vec<(String, String)>, qry: HashMap<String, Value>, body: Vec<u8>) {
    logger::init();
    let collection_name = qry.get("collection_name").unwrap().as_str().unwrap();
    let vector_size = qry.get("vector_size").unwrap().as_str().unwrap();
    let vector_size: u64 = vector_size.parse().unwrap();
    let mut id: u64 = 0;

    if qry.contains_key("reset") {
        log::debug!("Reset the database");
        // Delete collection, ignore any error
        _ = delete_collection(collection_name).await;
        // Create collection
        let p = CollectionCreateParams { vector_size: vector_size };
        if let Err(e) = create_collection(collection_name, &p).await {
            log::error!("Cannot create collection named: {} with error: {}", collection_name, e);
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

    let str_vec_received = serde_json::from_slice::<Vec<String>>(&body).unwrap_or(vec![]);

    let need_summarize_before_embed = str_vec_received.iter().any(|l| l.len() > CHAR_SOFT_LIMIT);

    let mut s = str_vec_received.clone();

    if need_summarize_before_embed {
        s.clear();
        for window in str_vec_received {
            let sum = summarize_long_chunks(&window).await;
            s.push(sum);
        }
    }

    let mut points = Vec::<Point>::new();
    for line in s {
        let input = EmbeddingsInput::String(line.clone());
        match openai.create_embeddings(input).await {
            Ok(r) => {
                for v in r.iter() {
                    let p = Point {
                        id: PointId::Num(id),
                        vector: v
                            .iter()
                            .map(|n| *n as f32)
                            .collect(),
                        payload: json!({"text": line}).as_object().map(|m| m.to_owned()),
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
            send_success(
                &format!(
                    "Successfully inserted {} records. The collection now has {} records in total.",
                    points_count,
                    ci.points_count
                )
            );
        }
        Err(e) => {
            log::error!("Cannot get collection stat {}", e);
            send_success("Cannot upsert into database!");
        }
    }
}

fn send_success(body: &str) {
    send_response(
        200,
        vec![(String::from("content-type"), String::from("text/html"))],
        body.as_bytes().to_vec()
    );
}

pub async fn summarize_long_chunks(input: &str) -> String {
    let sys_prompt_1 = format!("You're a technical edtior bot.");
    let co = chat::ChatOptions {
        model: chat::ChatModel::GPT35Turbo16K,
        system_prompt: Some(&sys_prompt_1),
        restart: true,
        temperature: Some(0.7),
        max_tokens: Some(256),
        ..Default::default()
    };
    let usr_prompt_1 = format!(
        "To prepare for downstream question & answer task, you need to proprocess the source material, there are long chunks of text that are tool long to use as context, you need to extract the essence of such chunks, now please summarize this chunk: `{input}` into one concise paragraph, please stay truthful to the source material and handle the task in a factual manner."
    );

    let mut openai = OpenAIFlows::new();
    openai.set_retry_times(2);

    match openai.chat_completion("summarize-long-chunks", &usr_prompt_1, &co).await {
        Ok(r) => r.choice,

        Err(_e) => "".to_owned(),
    }
}
