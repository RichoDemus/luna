use bevy::prelude::*;
use bevy::window::PrimaryWindow;

pub struct MouseCoordinatesPlugin;

impl Plugin for MouseCoordinatesPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup);
        app.add_systems(Update, text_follows_mouse);
    }
}

#[derive(Component)]
struct MouseCoordinates;

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
                       MouseCoordinates,
        Text::new("eyy"),
        TextFont {
            font: asset_server.load("fonts/orbiton/Orbitron-VariableFont_wght.ttf"),
            font_size: 14.0,
            ..default()
        },
        TextShadow::default(),
        // TextLayout::new_with_justify(Justify::Center),
        Node {
            position_type: PositionType::Absolute,
            bottom: px(5),
            right: px(5),
            ..default()
        },
    ));
}

fn text_follows_mouse(
    mut evr_cursor: MessageReader<CursorMoved>,
    mut texts: Query<(&mut Node, &mut Text), With<MouseCoordinates>>,
    q_window: Query<&Window, With<PrimaryWindow>>,
    q_camera: Query<(&Camera, &GlobalTransform)>,
) {
    for ev in evr_cursor.read() {
        for (mut node, mut span) in texts.iter_mut() {
            node.top = px(ev.position.y + 5.);
            node.left = px(ev.position.x+ 5. );

            let (camera, camera_transform) = q_camera.single().unwrap();

            let window = q_window.single().unwrap();

            if let Some(world_position) = window.cursor_position()
                .and_then(|cursor| camera.viewport_to_world(camera_transform, cursor).ok())
                .map(|ray| ray.origin.truncate())
            {
                eprintln!("World coords: {}/{}", world_position.x, world_position.y);
                **span = format!("{:.0}, {:.0}", world_position.x, world_position.y);
            }


        }
    }
}
