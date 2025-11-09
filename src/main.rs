mod core;
mod mouse_coordinates_plugin;
mod persistence;
mod q;
mod types;

use crate::core::LanderEnv;
use crate::mouse_coordinates_plugin::MouseCoordinatesPlugin;
use crate::q::QLearning;
use bevy::prelude::*;
use bevy::window::{WindowResolution, WindowTheme};
use once_cell::sync::Lazy;
use std::f32::consts::FRAC_PI_2;
use std::sync::Mutex;

pub(crate) const MIN_HEIGHT: f32 = 0.0;
pub(crate) const MAX_HEIGHT: f32 = 150.0;
pub(crate) const MIN_VELOCITY: f32 = -20.0;
pub(crate) const MAX_VELOCITY: f32 = 50.0;
pub(crate) const HEIGHT_BINS: usize = 120;
pub(crate) const VELOCITY_BINS: usize = 120;
pub(crate) const NUMBER_OF_ACTIONS: usize = 2;
pub(crate) const EPISODES: usize = 500_000;
pub(crate) const MAX_STEPS_PER_EPISODE: usize = 2_000usize;
pub(crate) const WINDOW_WIDTH: u32 = 1280;
pub(crate) const WINDOW_HEIGHT: u32 = 720;

fn main() {
    // core::run();
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                resolution: (WINDOW_WIDTH, WINDOW_HEIGHT).into(),
                window_theme: Some(WindowTheme::Dark),
                ..default()
            }),
            ..default()
        }))
        // .add_plugins(MouseCoordinatesPlugin)
        .add_systems(Startup, setup)
        // .add_systems(Startup, create_layout)
        .add_systems(Startup, spawn_stuff)
        .add_systems(Startup, place_camera.after(setup))
        .add_systems(FixedUpdate, just_do_everything)
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(Camera2d);
    commands.spawn((
        StateText,
        Text::new("asd"),
        TextFont {
            font: asset_server.load("fonts/roboto_mono/static/RobotoMono-Regular.ttf"),
            font_size: 20.0,
            ..default()
        },
        TextShadow::default(),
        Node {
            position_type: PositionType::Absolute,
            bottom: px(5),
            left: px(15),
            ..default()
        },
    ));
}

fn place_camera(mut camera: Single<&mut Transform, With<Camera>>) {
    camera.translation.y = 300.;
}

#[derive(Component)]
struct StateText;

#[derive(Component)]
struct Lander;

#[derive(Component, PartialEq, Eq, Debug, Copy, Clone)]
enum LanderState {
    Falling,
    Landed,
    Crashed,
}

#[derive(Component)]
struct Surface;

fn spawn_stuff(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<ColorMaterial>>) {
    let mesh = commands
        .spawn((
            Mesh2d(meshes.add(Triangle2d::new(
                Vec2::Y * 32.,
                Vec2::new(-32., -32.),
                Vec2::new(32., -32.),
            ))),
            MeshMaterial2d(materials.add(Color::hsv(209., 0.2, 0.47))),
            Transform::from_xyz(0., 32., 0.),
        ))
        .id();

    commands
        .spawn((Lander, LanderState::Falling, Transform::from_xyz(0., 200., 0.)))
        .add_child(mesh);

    let mesh = commands
        .spawn((
            Mesh2d(meshes.add(Rectangle::new(100., 20.))),
            MeshMaterial2d(materials.add(Color::hsv(59., 0.1, 0.8))),
            Transform::from_xyz(0., -10., 0.),
        ))
        .id();

    commands
        .spawn((Surface, Transform::from_xyz(0., 0., 0.)))
        .add_child(mesh);
}

static LANDER: Lazy<Mutex<LanderEnv>> = Lazy::new(|| {
    let env = LanderEnv::new(12345_u64 ^ 0xBEEF);
    Mutex::new(env)
});

static Q_LEARNING: Lazy<Mutex<QLearning>> = Lazy::new(|| {
    let mut q_learning = QLearning::new(12345_u64 ^ 0xDEAD_BEEF);
    q_learning.table = persistence::load().unwrap();
    Mutex::new(q_learning)
});

fn just_do_everything(
    mut landers: Query<(&mut Transform, &mut LanderState), With<Lander>>,
    mut texts: Query<&mut Text, With<StateText>>,
) {
    for (mut transform, mut state) in landers.iter_mut() {
        if *state != LanderState::Falling {
            continue;
        }
        let mut env = LANDER.lock().unwrap();
        let mut q_learning = Q_LEARNING.lock().unwrap();

        let discretized_height = env.state.height.discretize();
        let discretized_velocity = env.state.velocity.discretize();
        let (action, _) = q_learning.get_greedy_action_and_q_value(discretized_height, discretized_velocity);

        let (s_next, terminal_reward, done, _fuel) = env.step(action);
        if done {
            if terminal_reward > 0. {
                *state = LanderState::Landed;
                println!("landed");
                info!("height: {}, velocity: {}", s_next.height.0, s_next.velocity.0);
                for mut text in texts.iter_mut() {
                    **text = format!(
                        "State: {:>8?} | Velocity: {:>7.1} | Height: {:>7.1} | Thrusting: {:>5}",
                        *state,
                        s_next.velocity.0,
                        s_next.height.0,
                        action == 1
                    )
                }
            } else {
                *state = LanderState::Crashed;
                println!("crashed");
                info!("height: {}, velocity: {}", s_next.height.0, s_next.velocity.0);
                for mut text in texts.iter_mut() {
                    **text = format!(
                        "State: {:>8?} | Velocity: {:>7.1} | Height: {:>7.1} | Thrusting: {:>5}",
                        *state,
                        s_next.velocity.0,
                        s_next.height.0,
                        action == 1
                    )
                }
            }
            continue;
        }
        transform.translation.y = s_next.height.0 * 5.;
        info!("height: {}, velocity: {}", s_next.height.0, s_next.velocity.0);
        for mut text in texts.iter_mut() {
            **text = format!(
                "State: {:>8?} | Velocity: {:>7.1} | Height: {:>7.1} | Thrusting: {:>5}",
                *state,
                s_next.velocity.0,
                s_next.height.0,
                action == 1
            )
        }
    }
}
