use crate::q::QTable;

#[cfg(not(test))]
const PATH: &str = "q_table.json";
#[cfg(test)]
const PATH: &str = "target/q_table.json";

pub(crate) fn save(table: &QTable) {
    let vector: Vec<Vec<Vec<f32>>> = table.iter().map(|a| a.iter().map(|b| b.to_vec()).collect()).collect();
    let bytes = serde_json::to_vec(&vector).unwrap();
    std::fs::write(PATH, bytes).unwrap();
}

// pub(crate) enum LoadSource {
//     Root, Assets, Target,
// }

pub(crate) fn load() -> Option<QTable> {
    // let byte: Vec<u8> = match source {
    //     LoadSource::Root => {}
    //     LoadSource::Assets => include_bytes!("../assets/q_table.json").to_vec(),
    //     LoadSource::Target => {}
    // };
    // let bytes = include_bytes!("../q_table.json").to_vec();
    let bytes = std::fs::read(PATH).ok()?;
    let result: Vec<Vec<Vec<f32>>> = serde_json::from_slice(bytes.as_slice()).ok()?;

    let table: QTable = std::array::from_fn(|height| {
        std::array::from_fn(|velocity| std::array::from_fn(|action| result[height][velocity][action]))
    });

    Some(table)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::q::{NUMBER_OF_ACTIONS, QTable, VELOCITY_BINS};

    #[test]
    fn test_save_and_load() {
        if std::env::var("GITHUB_ACTIONS").as_deref() == Ok("true") {
            return;
        }
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
