extern crate reqwest;
extern crate serde;
extern crate serde_json;

use reqwest::blocking::*;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde_json::Map;
use serde_json::Value;
use std::collections::HashMap;
use std::error::Error;
use std::result::Result;

pub struct GymClient {
    base_uri: String,
    client: Client,
}

pub type GymResult<T> = Result<T, Box<dyn Error>>;

#[derive(Debug, Clone)]
pub enum Space {
    Discrete {
        n: u64,
    },

    Box {
        shape: Vec<u64>,
        high: Vec<f64>,
        low: Vec<f64>,
    },

    Tuple {
        spaces: Vec<Box<Space>>,
    },
}

impl Space {
    pub fn from_json(info: &Map<String, Value>) -> GymResult<Self> {
        match info["name"].as_str().ok_or("No name returned.")? {
            "Discrete" => {
                let n = info["n"].as_str().unwrap().parse::<u64>().unwrap();
                Ok(Space::Discrete { n })
            }
            "Box" => {
                let shape = info["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap())
                    .collect::<Vec<_>>();

                let high = info["high"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_str().unwrap().parse::<f64>().unwrap())
                    .collect::<Vec<_>>();

                let low = info["low"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_str().unwrap().parse::<f64>().unwrap())
                    .collect::<Vec<_>>();

                Ok(Space::Box { shape, high, low })
            }
            "Tuple" => panic!("Parsing for Tuple spaces is not yet implemented"),
            e => panic!("Unrecognized space name: {}", e),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Environment {
    instance_id: String,
    obs_space: Space,
    act_space: Space,
}

impl Environment {
    pub fn new(instance_id: &str, act_space: Space, obs_space: Space) -> Self {
        Self {
            instance_id: instance_id.to_string(),
            act_space,
            obs_space,
        }
    }

    pub fn instance_id(&self) -> &str {
        &self.instance_id
    }

    pub fn action_space(&self) -> &Space {
        &self.act_space
    }

    pub fn observation_space(&self) -> &Space {
        &self.obs_space
    }
}

impl GymClient {
    pub fn new(host: &str, port: u16) -> Self {
        Self {
            base_uri: format!("{host}:{port}"),
            client: Client::builder().build().unwrap(),
        }
    }

    pub fn get_envs(&self) -> Result<HashMap<String, String>, Box<dyn Error>> {
        let url = self.construct_req_url("/v1/envs/");
        let res = self.http_get(&url)?;

        let val = res.json::<Value>()?;
        let obj = val["all_envs"].as_object().ok_or("No all_envs returned.")?;
        let ret: HashMap<_, _> = obj
            .into_iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
            .collect();

        Ok(ret)
    }

    pub fn make_env(&self, env_id: &str) -> GymResult<Environment> {
        let body = HashMap::from([("env_id", env_id.to_string())]);

        let base_url = self.construct_req_url("/v1/envs/");
        let res = self
            .client
            .post(&base_url)
            .headers(Self::construct_common_headers())
            .json(&body)
            .send();

        let inst_id = res?.json::<Value>()?;
        let inst_id = inst_id["instance_id"].as_str().unwrap();

        let url = format!("{base_url}{inst_id}/observation_space/");
        let res = self.http_get(&url);
        let ret = res?.json::<Value>()?;
        let info = ret["info"].as_object().unwrap();
        let obs_space = Space::from_json(info);

        let url = format!("{base_url}{inst_id}/action_space/");
        let res = self.http_get(&url);
        let ret = res?.json::<Value>()?;
        let info = ret["info"].as_object().unwrap();
        let act_space = Space::from_json(info);

        Ok(Environment::new(inst_id, obs_space?, act_space?))
    }

    fn construct_req_url(&self, path: &str) -> String {
        format!("{}{}", self.base_uri, path)
    }

    fn http_get(&self, url: &str) -> GymResult<Response> {
        let res = self
            .client
            .get(url)
            .headers(Self::construct_common_headers())
            .send();

        Ok(res?)
    }

    fn construct_common_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers
    }

    pub fn reset_env(&self) {
        // TODO: env.reset with seed
        unimplemented!()
    }
}
