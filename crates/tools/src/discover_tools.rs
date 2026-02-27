//! `discover_tools` — meta-tool for lazy tool injection.
//!
//! Returns matching tool schemas so the LLM can discover and use tools
//! that aren't in the default "core" set.  The agent runner adds
//! discovered schemas to subsequent API calls automatically.

use {
    anyhow::Result,
    async_trait::async_trait,
    moltis_agents::tool_registry::AgentTool,
    serde::Deserialize,
    tracing::debug,
};

#[derive(Debug, Deserialize)]
struct Params {
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    10
}

/// Tool that searches all available tool schemas by keyword.
///
/// Holds a snapshot of all tool schemas at construction time.
/// The agent runner detects `discover_tools` results and injects
/// the returned schemas into the next LLM API call.
pub struct DiscoverToolsTool {
    all_schemas: Vec<serde_json::Value>,
}

impl DiscoverToolsTool {
    pub fn new(all_schemas: Vec<serde_json::Value>) -> Self {
        Self { all_schemas }
    }
}

#[async_trait]
impl AgentTool for DiscoverToolsTool {
    fn name(&self) -> &str {
        "discover_tools"
    }

    fn description(&self) -> &str {
        "Search for available tools by keyword. Returns matching tool schemas \
         that become available for use in subsequent turns. Use this when you \
         need a capability not in your current toolset — e.g. 'neo4j', \
         'browser', 'calendar', or 'map'."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords to search tool names and descriptions"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of tools to return (default: 10)",
                    "default": 10
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value> {
        let params: Params = serde_json::from_value(params)?;
        let query_lower = params.query.to_lowercase();
        let keywords: Vec<&str> = query_lower.split_whitespace().collect();

        let mut scored: Vec<(usize, &serde_json::Value)> = self
            .all_schemas
            .iter()
            .filter_map(|schema| {
                let name = schema.get("name")?.as_str()?.to_lowercase();
                let desc = schema
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or("")
                    .to_lowercase();
                let haystack = format!("{name} {desc}");

                // Score: count how many query keywords match.
                let hits = keywords
                    .iter()
                    .filter(|kw| haystack.contains(**kw))
                    .count();

                if hits > 0 {
                    Some((hits, schema))
                } else {
                    None
                }
            })
            .collect();

        // Sort by relevance (more keyword hits first), stable for equal scores.
        scored.sort_by(|a, b| b.0.cmp(&a.0));

        let results: Vec<&serde_json::Value> = scored
            .into_iter()
            .take(params.limit)
            .map(|(_, schema)| schema)
            .collect();

        debug!(
            query = %params.query,
            matches = results.len(),
            total = self.all_schemas.len(),
            "discover_tools search"
        );

        Ok(serde_json::json!({
            "tools": results,
            "total_available": self.all_schemas.len(),
            "message": format!(
                "Found {} tool(s) matching '{}'. These tools are now available for use.",
                results.len(),
                params.query
            )
        }))
    }
}

#[allow(clippy::unwrap_used, clippy::expect_used)]
#[cfg(test)]
mod tests {
    use super::*;

    fn sample_schemas() -> Vec<serde_json::Value> {
        vec![
            serde_json::json!({
                "name": "exec",
                "description": "Execute a shell command",
                "parameters": {}
            }),
            serde_json::json!({
                "name": "mcp__neo4j__read_cypher",
                "description": "Run a Cypher query against the Neo4j database",
                "parameters": {}
            }),
            serde_json::json!({
                "name": "browser",
                "description": "Open a URL in a headless browser and return content",
                "parameters": {}
            }),
            serde_json::json!({
                "name": "web_search",
                "description": "Search the web using Brave or Perplexity",
                "parameters": {}
            }),
        ]
    }

    #[tokio::test]
    async fn finds_matching_tools() {
        let tool = DiscoverToolsTool::new(sample_schemas());
        let result = tool
            .execute(serde_json::json!({"query": "neo4j"}))
            .await
            .unwrap();

        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "mcp__neo4j__read_cypher");
    }

    #[tokio::test]
    async fn multi_keyword_scoring() {
        let tool = DiscoverToolsTool::new(sample_schemas());
        let result = tool
            .execute(serde_json::json!({"query": "search web"}))
            .await
            .unwrap();

        let tools = result["tools"].as_array().unwrap();
        // web_search matches both keywords, browser matches only via "content"
        assert!(!tools.is_empty());
        assert_eq!(tools[0]["name"], "web_search");
    }

    #[tokio::test]
    async fn respects_limit() {
        let tool = DiscoverToolsTool::new(sample_schemas());
        let result = tool
            .execute(serde_json::json!({"query": "a", "limit": 1}))
            .await
            .unwrap();

        let tools = result["tools"].as_array().unwrap();
        assert!(tools.len() <= 1);
    }

    #[tokio::test]
    async fn no_matches_returns_empty() {
        let tool = DiscoverToolsTool::new(sample_schemas());
        let result = tool
            .execute(serde_json::json!({"query": "nonexistent_xyz"}))
            .await
            .unwrap();

        let tools = result["tools"].as_array().unwrap();
        assert!(tools.is_empty());
    }

    #[tokio::test]
    async fn returns_total_available() {
        let tool = DiscoverToolsTool::new(sample_schemas());
        let result = tool
            .execute(serde_json::json!({"query": "exec"}))
            .await
            .unwrap();

        assert_eq!(result["total_available"], 4);
    }
}
