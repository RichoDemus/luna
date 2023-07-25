use rurel::{AgentTrainer, mdp};
use rurel::mdp::Agent;
use rurel::strategy::explore::RandomExploration;
use rurel::strategy::learn::QLearning;
use rurel::strategy::terminate::FixedIterations;

// #[derive(PartialEq, Eq, Hash, Clone, Debug)]
// pub struct MyState {
//     pub height: BigDecimal,
//     pub velocity: BigDecimal,
//     pub fuel: BigDecimal,
//     pub status: ShipStatus,
//     // pub thrusting: bool,
// }


impl mdp::State for MyState {
    type A = MyAction;

    fn reward(&self) -> f64 {
        // should this be improved?
        return if self.status == Landed {
            self.fuel.to_f64().expect("couldn't convert fuel to f64")
        } else if self.status == Crashed {
            -100.
        } else {
            0.
        };
    }

    fn actions(&self) -> Vec<MyAction> {
        vec![
            DoNothing, Thrust,
        ]
    }
}


pub struct MyAgent {
    state: MyState,
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum MyAction {
    DoNothing,
    Thrust,
}

impl Agent<MyState> for MyAgent {
    fn current_state(&self) -> &MyState {
        &self.state
    }

    fn take_action(&mut self, action: &MyAction) {
        self.state = do_a_whole_fucking_tick(&self.state, action);
    }
}

pub fn do_a_whole_fucking_tick(state: &MyState, action: &MyAction) -> MyState {
    let ship_height = state.height.to_f32().unwrap();
    let velocity = state.velocity.to_f32().unwrap();
    let status = state.status;
    let fuel = state.fuel.to_f32().unwrap();
    let create_ship2 = move |mut commands: Commands| {
        commands.spawn((
            Ship,
            status,
            FuelTank(fuel),
            Transform::from_xyz(0., ship_height, 10.),
            Velocity(Vec2::new(0., velocity)),
        ));
    };

    let thrusting = action == &Thrust;
    let input2 = move |time: Res<Time>, mut ships: Query<(&mut Velocity, &ShipStatus, &mut FuelTank)>| {
        if thrusting {
            for (mut velocity, status, mut fuel) in &mut ships {
                if status != &ShipStatus::Falling {
                    continue;
                }
                velocity.0.y += 50. * time.delta_seconds();
                fuel.0 -= 10. * time.delta_seconds();
            }
        }
    };

    let mut app = App::new();

    app.add_plugins((TaskPoolPlugin::default(), TypeRegistrationPlugin::default(), FrameCountPlugin::default(), ScheduleRunnerPlugin::default()));
    app.add_systems(Startup, (create_ship2, create_ground));
    app.add_systems(Update, (input2, gravity.after(input2), movement.after(gravity), touchdown));
    // let mut time = app.world.resource_mut::<Time>();
    // time.update();
    // println!("{}", time.elapsed_seconds());
    app.world.init_resource::<Time>();
    let mut time = app.world.resource_mut::<Time>();
    time.update();
    app.update();

    // for (velocity, transform, status, fuel) in app.world.query::<(&Velocity, &Transform, &ShipStatus, &FuelTank)>().iter(&app.world) {
    //     println!(
    //         "Ship: {:?}: Pos: {:?} velocity: {:?}. {:?} fuel left",
    //         status, transform.translation, velocity, fuel
    //     );
    // }

    let mut time = app.world.resource_mut::<Time>();
    let last_update = time.last_update().unwrap();
    time.update_with_instant(last_update + Duration::from_millis(6));
    // println!("{}", time.elapsed_seconds());

    app.update();

    for (velocity, transform, status, fuel) in app.world.query::<(&Velocity, &Transform, &ShipStatus, &FuelTank)>().iter(&app.world) {
        // println!(
        //     "Ship: {:?}: Pos: {:?} velocity: {:?}. {:?} fuel left",
        //     status, transform.translation, velocity, fuel
        // );
        return MyState {
            height: BigDecimal::from_f32(transform.translation.y).unwrap(),
            velocity: BigDecimal::from_f32(velocity.0.y).unwrap(),
            fuel: BigDecimal::from_f32(fuel.0).unwrap(),
            status: *status,
            // thrusting,
        };
    }

    todo!()
}