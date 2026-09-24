//! The §2.6 error envelope, used by every non-2xx HTTP response.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
    pub param: Option<&'static str>,
}

impl ApiError {
    pub fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
            param: None,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        // OpenAI uses `invalid_request_error` for client errors and
        // `server_error` for server errors.
        let kind = if self.status.is_server_error() {
            "server_error"
        } else {
            "invalid_request_error"
        };
        let body = json!({"error": {
            "message": self.message,
            "type": kind,
            "code": self.code,
            "param": self.param,
        }});
        (self.status, Json(body)).into_response()
    }
}
