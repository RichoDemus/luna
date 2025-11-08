use crate::q::QTable;
use crate::{NUMBER_OF_ACTIONS, VELOCITY_BINS};

#[cfg(not(test))]
const PATH: &'static str = "q_table.json";
#[cfg(test)]
const PATH: &'static str = "target/q_table.json";

pub(crate) fn save(table: &QTable) {
    let vector: Vec<Vec<Vec<f32>>> = table
        .into_iter()
        .map(|a| a.into_iter().map(|b| b.to_vec()).collect())
        .collect();
    let bytes = serde_json::to_vec(&vector).unwrap();
    std::fs::write(PATH, bytes).unwrap();
}

pub(crate) fn load() -> Option<QTable> {
    let bytes = include_bytes!("../q_table.json").to_vec();
    // let bytes = std::fs::read(PATH).ok()?;
    let result: Vec<Vec<Vec<f32>>> = serde_json::from_slice(bytes.as_slice()).ok()?;

    let table: QTable = std::array::from_fn(|height| {
        std::array::from_fn(|velocity| {
            std::array::from_fn(|action| {
                result
                    .get(height)
                    .unwrap()
                    .get(velocity)
                    .unwrap()
                    .get(action)
                    .unwrap()
                    .clone()
            })
        })
    });

    Some(table)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::q::QTable;

    #[test]
    fn test_save_and_load() {
        let table: QTable = std::array::from_fn(|height| {
            std::array::from_fn(|velocity| {
                std::array::from_fn(|action| {
                    (height * VELOCITY_BINS * 2 + velocity * NUMBER_OF_ACTIONS + action) as f32
                })
            })
        });

        save(&table);
        let loaded_table = load().unwrap();

        assert_eq!(table, loaded_table);
    }
}
