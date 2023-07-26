use bevy::prelude::*;

use crate::lander::{Altitude, FuelTank, Lander, ShipStatus, Thruster, Velocity};

pub(crate) struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup);
        app.add_systems(Update, update_text);
    }
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    // root node
    commands
        .spawn(NodeBundle {
            style: Style {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::SpaceBetween,
                ..default()
            },
            ..default()
        })
        .with_children(|parent| {
            // left vertical fill (border)
            parent
                .spawn(NodeBundle {
                    style: Style {
                        width: Val::Px(200.),
                        border: UiRect::all(Val::Px(2.)),
                        ..default()
                    },
                    background_color: Color::rgb(0.65, 0.65, 0.65).into(),
                    ..default()
                })
                .with_children(|parent| {
                    // left vertical fill (content)
                    parent
                        .spawn(NodeBundle {
                            style: Style {
                                width: Val::Percent(100.),
                                ..default()
                            },
                            background_color: Color::rgb(0.15, 0.15, 0.15).into(),
                            ..default()
                        })
                        .with_children(|parent| {
                            // text
                            parent.spawn((
                                TextBundle::from_section(
                                    "Text Example",
                                    TextStyle {
                                        font: asset_server.load("fonts/FiraMono-Medium.ttf"),
                                        font_size: 20.0,
                                        color: Color::WHITE,
                                    },
                                )
                                .with_style(Style {
                                    margin: UiRect::all(Val::Px(5.)),
                                    ..default()
                                }),
                                // Because this is a distinct label widget and
                                // not button/list item text, this is necessary
                                // for accessibility to treat the text accordingly.
                                Label,
                                InfoBox,
                            ));
                        });
                });
        });
}

#[derive(Component)]
struct InfoBox;

#[derive(Component, Default)]
struct ScrollingList {
    position: f32,
}

fn update_text(
    mut text_box: Query<&mut Text, With<InfoBox>>,
    lander: Query<(&Altitude, &Velocity, &FuelTank, &ShipStatus, &Thruster), With<Lander>>,
) {
    let (altitude, velocity, fuel, status, thruster) = lander.single();
    let mut text = text_box.single_mut();

    text.sections[0].value = format!(
        "Altitude: {}m\nVelocity: {}m/s\nFuel: {}\nStatus: {:?}\nThrust: {}",
        altitude.0 / 1000,
        velocity.0 / 1000,
        fuel.0,
        status,
        thruster.0,
    );
}
