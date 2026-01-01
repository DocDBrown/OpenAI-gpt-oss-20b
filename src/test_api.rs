use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::fmt;

use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, header, method, path},
};

#[derive(Serialize, Debug, Clone)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: Vec<Message<'a>>,
    stream: bool,
    temperature: f32,
}

#[derive(Serialize, Debug, Clone)]
struct Message<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize, Debug)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize, Debug)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Deserialize, Debug)]
struct ChoiceMessage {
    content: String,
}

#[derive(Debug)]
enum ApiError {
    HttpStatus { status: u16, body: String },
    Transport(String),
    JsonParse { err: String, raw: String },
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApiError::HttpStatus { status, body } => write!(f, "HTTP {}: {}", status, body),
            ApiError::Transport(e) => write!(f, "Transport error: {}", e),
            ApiError::JsonParse { err, raw } => {
                write!(f, "JSON parse error: {}\nRaw: {}", err, raw)
            }
        }
    }
}

impl std::error::Error for ApiError {}

fn endpoint_for(base_url: &str) -> String {
    format!("{}/v1/chat/completions", base_url.trim_end_matches('/'))
}

async fn call_chat_completion(
    base_url: &str,
    model: &str,
    prompt: &str,
) -> Result<String, ApiError> {
    let endpoint = endpoint_for(base_url);

    let req_body = ChatCompletionRequest {
        model,
        messages: vec![
            Message {
                role: "system",
                content: "You are a helpful assistant.",
            },
            Message {
                role: "user",
                content: prompt,
            },
        ],
        stream: false,
        temperature: 0.2,
    };

    let client = Client::new();
    let resp = client
        .post(&endpoint)
        .json(&req_body)
        .send()
        .await
        .map_err(|e| ApiError::Transport(e.to_string()))?;

    let status = resp.status().as_u16();
    let text = resp
        .text()
        .await
        .map_err(|e| ApiError::Transport(e.to_string()))?;

    if !(200..300).contains(&status) {
        return Err(ApiError::HttpStatus { status, body: text });
    }

    let parsed: ChatCompletionResponse =
        serde_json::from_str(&text).map_err(|e| ApiError::JsonParse {
            err: e.to_string(),
            raw: text.clone(),
        })?;

    // Clippy prefers .first() over .get(0). :contentReference[oaicite:2]{index=2}
    Ok(parsed
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .unwrap_or_default())
}

#[tokio::test]
async fn endpoint_builder_trims_trailing_slash() {
    assert_eq!(
        endpoint_for("http://127.0.0.1:8080/"),
        "http://127.0.0.1:8080/v1/chat/completions"
    );
    assert_eq!(
        endpoint_for("http://127.0.0.1:8080"),
        "http://127.0.0.1:8080/v1/chat/completions"
    );
}

#[tokio::test]
async fn happy_path_returns_assistant_content_and_sends_expected_payload() {
    let server = MockServer::start().await;

    let expected_body = serde_json::json!({
        "model": "gpt-oss-120b",
        "messages": [
            { "role": "system", "content": "You are a helpful assistant." },
            { "role": "user", "content": "Say hello." }
        ],
        "stream": false,
        "temperature": 0.2
    });

    let response_body = serde_json::json!({
        "choices": [
            { "message": { "content": "Hello." } }
        ]
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("content-type", "application/json"))
        .and(body_json(expected_body))
        .respond_with(ResponseTemplate::new(200).set_body_json(response_body))
        .mount(&server)
        .await;

    let out = call_chat_completion(&server.uri(), "gpt-oss-120b", "Say hello.")
        .await
        .expect("expected success");

    assert_eq!(out, "Hello.");
}

#[tokio::test]
async fn base_url_with_trailing_slash_still_works() {
    let server = MockServer::start().await;

    let response_body = serde_json::json!({
        "choices": [
            { "message": { "content": "OK" } }
        ]
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response_body))
        .mount(&server)
        .await;

    let base = format!("{}/", server.uri());
    let out = call_chat_completion(&base, "gpt-oss-120b", "ping")
        .await
        .expect("expected success");

    assert_eq!(out, "OK");
}

#[tokio::test]
async fn non_2xx_status_is_reported_with_body() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .mount(&server)
        .await;

    let err = call_chat_completion(&server.uri(), "gpt-oss-120b", "ping")
        .await
        .expect_err("expected failure");

    match err {
        ApiError::HttpStatus { status, body } => {
            assert_eq!(status, 500);
            assert_eq!(body, "boom");
        }
        other => panic!("unexpected error variant: {:?}", other),
    }
}

#[tokio::test]
async fn invalid_json_body_is_reported_as_parse_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .mount(&server)
        .await;

    let err = call_chat_completion(&server.uri(), "gpt-oss-120b", "ping")
        .await
        .expect_err("expected failure");

    match err {
        ApiError::JsonParse { raw, .. } => {
            assert_eq!(raw, "not json");
        }
        other => panic!("unexpected error variant: {:?}", other),
    }
}

#[tokio::test]
async fn empty_choices_returns_empty_string() {
    let server = MockServer::start().await;

    let response_body = serde_json::json!({ "choices": [] });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response_body))
        .mount(&server)
        .await;

    let out = call_chat_completion(&server.uri(), "gpt-oss-120b", "ping")
        .await
        .expect("expected success");

    assert_eq!(out, "");
}

#[tokio::test]
async fn concurrency_multiple_requests_succeeds() {
    let server = MockServer::start().await;

    let response_body = serde_json::json!({
        "choices": [
            { "message": { "content": "OK" } }
        ]
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response_body))
        .expect(20)
        .mount(&server)
        .await;

    let mut handles = Vec::new();
    for _ in 0..20 {
        let base = server.uri();
        handles.push(tokio::spawn(async move {
            call_chat_completion(&base, "gpt-oss-120b", "ping").await
        }));
    }

    for h in handles {
        let res = h.await.expect("join");
        assert_eq!(res.expect("api ok"), "OK");
    }
}

#[tokio::test]
async fn invalid_base_url_is_reported_as_transport_error() {
    let err = call_chat_completion("http://:", "gpt-oss-120b", "ping")
        .await
        .expect_err("expected failure");

    match err {
        ApiError::Transport(_msg) => {}
        other => panic!("unexpected error variant: {:?}", other),
    }
}
