extern crate rand;
extern crate reqwest;
extern crate serde;
extern crate serde_json;

use rand::Rng;
use reqwest::{
    blocking::*,
    header::{HeaderMap, HeaderValue, CONTENT_TYPE},
};
use serde::ser::Serialize;
use serde_json::{to_value, Map, Value};
use std::collections::HashMap;
use std::error::Error;

#[derive(Debug, Clone)]
pub struct GymClient {
    base_uri: String,
    client: Client,
}

pub type GymResult<T> = Result<T, Box<dyn Error>>;

#[derive(Debug, Clone, Copy)]
pub enum ObsActSpaceItem {
    Discrete(usize),
    Box(f32),
}

impl ObsActSpaceItem {
    pub fn discrete_value(&self) -> usize {
        match self {
            Self::Discrete(n) => *n,
            _ => panic!("Is not a Discrete item"),
        }
    }

    pub fn box_value(&self) -> f32 {
        match self {
            Self::Box(n) => *n,
            _ => panic!("Is not a Discrete item"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ObsActSpace {
    Discrete {
        n: usize,
    },

    Box {
        shape: Vec<usize>,
        high: Vec<f32>,
        low: Vec<f32>,
    },

    Tuple {
        spaces: Vec<ObsActSpace>,
    },
}

impl ObsActSpace {
    pub fn from_json(info: &Map<String, Value>) -> GymResult<Self> {
        match info["name"].as_str().ok_or("No name returned.")? {
            "Discrete" => {
                let n = GymClient::value_to_number::<usize>(&info["n"]);
                Ok(ObsActSpace::Discrete { n })
            }
            "Box" => {
                let shape = GymClient::value_to_vec::<usize>(&info["shape"]);
                let high = GymClient::value_to_vec::<f32>(&info["high"]);
                let low = GymClient::value_to_vec::<f32>(&info["low"]);
                Ok(ObsActSpace::Box { shape, high, low })
            }
            "Tuple" => panic!("Parsing for Tuple spaces is not yet implemented"),
            e => panic!("Unrecognized space name: {}", e),
        }
    }

    pub fn items_from_json(&self, vals: &[Value]) -> Vec<ObsActSpaceItem> {
        match self {
            ObsActSpace::Discrete { n: _ } => vals
                .iter()
                .map(|v| ObsActSpaceItem::Discrete(v.as_u64().unwrap() as usize))
                .collect::<Vec<_>>(),

            ObsActSpace::Box {
                shape: _,
                high: _,
                low: _,
            } => vals
                .iter()
                .map(|v| ObsActSpaceItem::Box(v.as_f64().unwrap() as f32))
                .collect::<Vec<_>>(),

            ObsActSpace::Tuple { spaces: _ } => unimplemented!("Not yet implemented for tuples."),
        }
    }

    pub fn sample(&self) -> Vec<ObsActSpaceItem> {
        let mut rng = rand::thread_rng();
        match self {
            ObsActSpace::Discrete { n } => {
                vec![ObsActSpaceItem::Discrete(rng.gen::<usize>() % n)]
            }
            ObsActSpace::Box {
                ref shape,
                ref high,
                ref low,
            } => {
                let mut ret = Vec::with_capacity(shape.iter().copied().product());
                let mut index = 0;
                for &i in shape {
                    for _ in 0..i {
                        ret.push(ObsActSpaceItem::Box(rng.gen_range(low[index]..high[index])));
                        index += 1;
                    }
                }
                ret
            }
            ObsActSpace::Tuple { ref spaces } => {
                let mut ret = Vec::new();
                for space in spaces {
                    ret.extend(space.sample());
                }
                ret
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Transition {
    pub next_state: usize,
    pub probability: f32,
    pub reward: f32,
    pub done: bool,
}

pub type Transitions = HashMap<(usize, usize), Vec<Transition>>;

#[derive(Debug)]
pub struct Environment {
    client: GymClient,
    name: String,
    instance_id: String,
    obs_space: ObsActSpace,
    act_space: ObsActSpace,
}

#[derive(Debug)]
pub struct StepInfo {
    pub observation: Vec<ObsActSpaceItem>,
    pub reward: f64,
    pub truncated: bool,
    pub terminated: bool,
    pub info: Value,
}

impl Environment {
    pub fn new(
        client: GymClient,
        name: &str,
        instance_id: &str,
        obs_space: ObsActSpace,
        act_space: ObsActSpace,
    ) -> Self {
        Self {
            client,
            name: name.to_string(),
            instance_id: instance_id.to_string(),
            obs_space,
            act_space,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn instance_id(&self) -> &str {
        &self.instance_id
    }

    /// The Space object corresponding to valid actions, all valid actions should be contained with the space.
    /// For example, if the action space is of type Discrete and gives the value Discrete(2), this means there
    /// are two valid discrete actions: 0 & 1.
    pub fn action_space(&self) -> &ObsActSpace {
        &self.act_space
    }

    /// The Space object corresponding to valid observations, all valid observations should be contained with
    /// the space. For example, if the observation space is of type Box and the shape of the object is (4,),
    /// this denotes a valid observation will be an array of 4 numbers. We can check the box bounds as well with attributes.
    pub fn observation_space(&self) -> &ObsActSpace {
        &self.obs_space
    }

    pub fn reset(&self, seed: Option<usize>) -> GymResult<Vec<ObsActSpaceItem>> {
        let mut body = HashMap::from([]);
        if let Some(seed) = seed {
            let _ = body.insert("seed", seed.to_string());
        }

        let url = format!(
            "{}{}/reset/",
            self.client.construct_req_url("/v1/envs/"),
            self.instance_id
        );
        let obj = self.client.http_post(&url, &body)?;
        let obs = obj["observation"].as_array().unwrap();
        let obs = ObsActSpace::items_from_json(&self.obs_space, obs);

        Ok(obs)
    }

    pub fn render(&self) -> GymResult<String> {
        let url = format!(
            "{}{}/render/",
            self.client.construct_req_url("/v1/envs/"),
            self.instance_id
        );
        let obj = self.client.http_get(&url)?;
        let render_frame = obj["render_frame"].as_str().unwrap();
        let render_frame = render_frame.replace("\\u", "\\x").replace(['{', '}'], "");
        Ok(render_frame)
    }

    pub fn step(&self, action: Vec<ObsActSpaceItem>) -> GymResult<StepInfo> {
        let mut req = HashMap::from([]);

        match self.act_space {
            ObsActSpace::Discrete { .. } => {
                if action.len() != 1 {
                    panic!("For Discrete space: Expected only one action.")
                }
                if let ObsActSpaceItem::Discrete(action) = action[0] {
                    req.insert("action", to_value(action).unwrap());
                } else {
                    panic!("For Discrete space: Expected only one action of type usize.")
                }
            }

            ObsActSpace::Box { ref shape, .. } => {
                if action.len() != shape[0] {
                    panic!("For Box space: Expected same number of actions as shape.")
                }
                let action: Vec<f32> = action
                    .iter()
                    .map(|a| {
                        if let ObsActSpaceItem::Box(a) = a {
                            *a
                        } else {
                            panic!("For Box space: Actions should all be f64")
                        }
                    })
                    .collect();
                req.insert("action", to_value(action).unwrap());
            }

            // TODO: This space thing is a bad design.
            ObsActSpace::Tuple { .. } => panic!("Actions for Tuple spaces not implemented yet"),
        }
        let url = format!(
            "{}{}/step/",
            self.client.construct_req_url("/v1/envs/"),
            self.instance_id
        );
        let obj = self.client.http_post(&url, &req)?;
        let observation = obj["observation"].as_array().unwrap();
        let observation = ObsActSpace::items_from_json(&self.obs_space, observation);

        Ok(StepInfo {
            observation,
            reward: obj["reward"].as_f64().unwrap(),
            truncated: obj["truncated"].as_bool().unwrap(),
            terminated: obj["terminated"].as_bool().unwrap(),
            info: obj["info"].clone(),
        })
    }

    pub fn transitions(&self) -> GymResult<Transitions> {
        let url = format!(
            "{}{}/transitions/",
            self.client.construct_req_url("/v1/envs/"),
            self.instance_id
        );
        let obj = self.client.http_get(&url)?;
        let obj = obj["transitions"].as_object().unwrap();

        let mut transitions: Transitions = HashMap::new();
        if let (ObsActSpace::Discrete { n: n_s }, ObsActSpace::Discrete { n: n_a }) =
            (self.observation_space(), self.action_space())
        {
            for s in 0..*n_s {
                let s_trans = obj[&s.to_string()].as_object().unwrap();
                for a in 0..*n_a {
                    let a_trans = s_trans[&a.to_string()].as_array().unwrap();
                    let ts = a_trans
                        .iter()
                        .map(|t| {
                            let t = t.as_array().unwrap();
                            Transition {
                                probability: t[0].as_f64().unwrap() as f32,
                                next_state: t[1].as_u64().unwrap() as usize,
                                reward: t[2].as_f64().unwrap() as f32,
                                done: t[3].as_bool().unwrap(),
                            }
                        })
                        .collect();

                    transitions.insert((s, a), ts);
                }
            }
        } else {
            panic!("Cannot get transition probabilities for environments that dont have discrete observation and action spaces.")
        }

        Ok(transitions)
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
        let val = self.http_get(&url)?;

        let obj = val["all_envs"].as_object().ok_or("No all_envs returned.")?;
        let ret: HashMap<_, _> = obj
            .into_iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
            .collect();

        Ok(ret)
    }

    pub fn make_env(self, env_id: &str, render_mode: Option<&str>) -> GymResult<Environment> {
        let mut body = HashMap::from([("env_id", env_id.to_string())]);
        if let Some(render_mode) = render_mode {
            body.insert("render_mode", render_mode.to_string());
        }

        let base_url = self.construct_req_url("/v1/envs/");
        let obj = self.http_post(&base_url, &body)?;
        let inst_id = obj["instance_id"].as_str().unwrap();

        let url = format!("{base_url}{inst_id}/observation_space/");
        let obj = self.http_get(&url)?;
        let info = obj["info"].as_object().unwrap();
        let obs_space = ObsActSpace::from_json(info);

        let url = format!("{base_url}{inst_id}/action_space/");
        let obj = self.http_get(&url)?;
        let info = obj["info"].as_object().unwrap();
        let act_space = ObsActSpace::from_json(info);

        Ok(Environment::new(
            self, env_id, inst_id, obs_space?, act_space?,
        ))
    }

    pub fn construct_req_url(&self, path: &str) -> String {
        format!("{}{}", self.base_uri, path)
    }

    fn value_to_number<T>(val: &Value) -> T
    where
        T: std::str::FromStr,
        <T as std::str::FromStr>::Err: std::fmt::Debug,
    {
        val.as_str().unwrap().parse::<T>().unwrap()
    }

    fn value_to_vec<T>(val: &Value) -> Vec<T>
    where
        T: std::str::FromStr,
        <T as std::str::FromStr>::Err: std::fmt::Debug,
    {
        val.as_array()
            .unwrap()
            .iter()
            .map(Self::value_to_number::<T>)
            .collect::<Vec<_>>()
    }

    fn http_get(&self, url: &str) -> GymResult<Value> {
        let res = self
            .client
            .get(url)
            .headers(Self::construct_common_headers())
            .send();
        let ret = res?.json::<Value>()?;

        Ok(ret)
    }

    fn http_post<T: Serialize>(&self, url: &str, body: &HashMap<&str, T>) -> GymResult<Value> {
        let res = self
            .client
            .post(url)
            .headers(Self::construct_common_headers())
            .json(body)
            .send();
        let ret = res?.json::<Value>()?;

        Ok(ret)
    }

    fn construct_common_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discrete_space_sample() {
        let n = 15;
        let discrete_space = ObsActSpace::Discrete { n };
        for _ in 0..10 {
            let sample = discrete_space.sample();
            assert!(sample.len() == 1);
            if let ObsActSpaceItem::Discrete(sample) = sample[0] {
                assert!(sample < n);
            } else {
                panic!("For discrete sample space samples must SampleItem::USize.");
            }
        }
    }

    #[test]
    fn test_box_space_sample() {
        let shape = vec![5];
        let high = vec![1., 2., 3., 4., 5.];
        let low = vec![-1., -2., -3., -4., -5.];
        let box_space = ObsActSpace::Box {
            shape: shape.clone(),
            high: high.clone(),
            low: low.clone(),
        };
        for _ in 0..10 {
            let sample = box_space.sample();
            assert_eq!(sample.len(), shape[0]);
            for i in 0..5 {
                if let ObsActSpaceItem::Box(sample) = sample[i] {
                    assert!(low[i] <= sample && sample <= high[i]);
                } else {
                    panic!("For discrete sample space samples must SampleItem::F32.");
                }
            }
        }
    }

    #[test]
    fn test_tuple_space_sample() {
        let n = 15;
        let discrete_space = ObsActSpace::Discrete { n };
        let shape = vec![5];
        let high = vec![1., 2., 3., 4., 5.];
        let low = vec![-1., -2., -3., -4., -5.];
        let box_space = ObsActSpace::Box {
            shape: shape.clone(),
            high: high.clone(),
            low: low.clone(),
        };

        let tuple_space = ObsActSpace::Tuple {
            spaces: vec![discrete_space, box_space],
        };
        for _ in 0..10 {
            let sample = tuple_space.sample();
            assert_eq!(sample.len(), shape[0] + 1);

            if let ObsActSpaceItem::Discrete(sample) = sample[0] {
                assert!(sample < n);
            } else {
                panic!("For discrete sample space samples must SampleItem::USize.");
            }

            for i in 1..6 {
                if let ObsActSpaceItem::Box(sample) = sample[i] {
                    assert!(low[i - 1] <= sample && sample <= high[i - 1]);
                } else {
                    panic!("For discrete sample space samples must SampleItem::F64.");
                }
            }
        }
    }
}
