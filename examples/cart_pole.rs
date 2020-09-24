extern crate cosyne;
extern crate sdl2;
extern crate random_color;


use sdl2::pixels::Color;
use sdl2::render::WindowCanvas;
use sdl2::gfx::primitives::DrawRenderer;
use sdl2::rect::Rect;
use sdl2::ttf::*;
use std::path::Path;
use sdl2::surface;
use std::time::Duration;
use std::thread;
use cosyne::activations::Activation;
use cosyne::network::ANN;
use cosyne::population::Population;
use cosyne::environment::Environment;
use cosyne::genome::Genome;


const MAX_STEP: usize = 50_000;

struct CartPoleEnvironment {}

pub struct Cart {
    pub width: f64,
    pub height: f64,
    pub color: Color,
    pub mass: f64,
    pub power: f64,
    pub pos: f64,
    pub vel: f64,
    pub acc: f64,
    pub track_friction: f64,

    pub pend_length: f64,
    pub pend_width: f64,
    pub pend_color: Color,
    pub ball_size: f64,
    pub pend_friction: f64,

    pub angle_pos: f64,
    pub angle_vel: f64,
    pub angle_acc: f64,

    pub gravity: f64,

    pub track_length: f64,
    pub window_width: f64,

    pub score: u32,
    pub dead: bool,
}

impl Cart {
    pub fn new(pend_length: f64, gravity: f64, mass: f64, power: f64, track_friction: f64, pend_friction: f64, track_length: f64, window_width: f64) -> Cart {
        // Pick Starting Angle
        let angle_pos = if rand::random() {
            0.001
        } else {
            -0.001
        };

        Cart {
            width: 30.0,
            height: 10.0,
            color: Color::from((255, 255, 255)),
            mass,
            power,
            pos: 0.0,
            vel: 0.0,
            acc: 0.0,
            track_friction,
            pend_length,
            pend_width: 2.0,
            pend_color: Color::from((255, 50, 0)),
            ball_size: 4.0,
            pend_friction,
            angle_pos,
            angle_vel: 0.0,
            angle_acc: 0.0,
            gravity,
            track_length,
            window_width,
            score: 0,
            dead: false,
        }
    }

    pub fn default() -> Cart {
        let track_length = 1200.0;
        let pend_length = 500.0;
        let gravity = 9.81;
        let cart_mass = 700.0;
        let cart_power = 12.0;
        let track_friction = 10.0;
        let pend_friction = 1.0;

        Cart::new(
            pend_length,
            gravity/1000.0,
            cart_mass/1000.0,
            cart_power/1000.0,
            track_friction/1000.0,
            pend_friction/1000.0,
            track_length,
            1280.0,
        )
    }

    pub fn update(&mut self, right_pressed: bool, left_pressed: bool) {
        // Keyboard Controls
        self.acc = 0.0;

        if right_pressed {
            self.acc += self.power;
        }

        if left_pressed {
            self.acc -= self.power;
        }

        self.acc -= self.track_friction * self.vel; // Add Track Friction

        // Cart Kinematics
        self.vel += self.acc;
        self.pos += self.vel;

        if self.pos >= self.track_length / 2.0 {
            self.pos = self.track_length / 2.0;
            self.vel = 0.0;
            self.acc = 0.0;
        }

        if self.pos <= -self.track_length / 2.0 {
            self.pos = -self.track_length / 2.0;
            self.vel = 0.0;
            self.acc = 0.0;
        }

        // Update Pendulum
        self.angle_acc = (-self.mass * self.acc * self.angle_pos.cos() + self.gravity * self.angle_pos.sin()) / self.pend_length;
        self.angle_acc -= self.pend_friction * self.angle_vel;

        self.angle_vel += self.angle_acc;
        self.angle_pos += self.angle_vel;

        // Update Score
        self.score += 1;

        if self.score >= 100_000 {
            // terminate on end condition
            self.dead = true;
        }

        if self.angle_pos >= std::f64::consts::PI/2.0 || self.angle_pos <= -std::f64::consts::PI/2.0 {
            self.dead = true;
        }
    }

    pub fn draw(&self, canvas: &mut WindowCanvas) {
        // Track
        let track_thickness = 2.0;
        let track_y = 600.0;
        let track_x1 = (self.window_width - self.track_length) / 2.0;
        // let track_x2 = track_x1 + self.track_length;
        let track_center = track_x1 + self.track_length / 2.0;

        // Cart Body
        let cart_x = track_center + self.pos - self.width/2.0;
        let cart_y = track_y - track_thickness/2.0 - self.height;
        canvas.set_draw_color(self.color);
        canvas.fill_rect(Rect::new(cart_x as i32, cart_y as i32, self.width as u32, self.height as u32)).unwrap();

        // Pendulum Pole
        let x1 = cart_x + self.width / 2.0;
        let y1 = cart_y;
        let dx = self.pend_length * self.angle_pos.sin();
        let dy = -self.pend_length * self.angle_pos.cos();
        let x2 = x1 + dx;
        let y2 = y1 + dy;

        canvas.thick_line(x1 as i16, y1 as i16, x2 as i16, y2 as i16, self.pend_width as u8, self.pend_color).unwrap();

        // Pendulum Ends
        canvas.filled_circle(x2 as i16, y2 as i16, self.ball_size as i16, self.pend_color).unwrap();
        canvas.filled_circle(x1 as i16, y1 as i16, self.pend_width as i16, self.color).unwrap();

    }
}

impl Environment for CartPoleEnvironment {
    fn evaluate(&self, nn: &mut ANN) -> f64 {
        let mut cart = Cart::default();
        while !cart.dead {
            if cart.score > MAX_STEP as u32 {
                break
            }

            let sensors: Vec<f64> = vec![
                cart.pos,
                cart.vel,
                cart.angle_pos,
                cart.angle_vel];

            let output = nn.forward(sensors);
            let right_press: bool = if output[0] > 0.8 {
                true
            } else {
                false
            };
            let left_press: bool = if output[0] < -0.8 {
                true
            } else {
                false
            };
            cart.update(right_press, left_press);
        }
        cart.score as f64
    }

    fn reset(&mut self) {}
}

fn main() {
    pretty_env_logger::init();
    cart_pole_feed_forward();
}

fn render_champion(mut champion: ANN) {
    // render video of champion
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let window_size = (1080, 1080);

    let window = video_subsystem.window(format!("Cart").as_str(), window_size.0, window_size.1)
        .position_centered()
        .build()
        .unwrap();
    let mut canvas = window.into_canvas()
        .accelerated()
        .build()
        .unwrap();

    // Initiate Text
    let ttf_ctx = sdl2::ttf::init().expect("Failed to init SDL_TTF");
    let font = Text::new(&ttf_ctx, "./res/anon.ttf", 20, Color::RGB(255, 255, 255)).expect("Failed to create font");

    let track_thickness = 2.0;
    let track_color = Color::from((255, 255, 255));
    let track_y = 600.0;

    let mut cart = Cart::default();

    while !cart.dead {
        if cart.score > MAX_STEP as u32 { break }

        let sensors: Vec<f64> = vec![
            cart.pos,
            cart.vel,
            cart.angle_pos,
            cart.angle_vel];

        let output = champion.forward(sensors);
        let right_press: bool = if output[0] > 0.8 {
            true
        } else {
            false
        };
        let left_press: bool = if output[0] < -0.8 {
            true
        } else {
            false
        };
        cart.update(right_press, left_press);

        canvas.set_draw_color(Color::from((0, 0, 0)));
        canvas.clear();
        // Draw Track
        canvas.thick_line(0, track_y as i16, window_size.0 as i16, track_y as i16, track_thickness as u8, track_color).unwrap();

        cart.draw(&mut canvas);
        font.draw_multi(&mut canvas, format!("SCORE: {}", cart.score).as_str(), 5, 20, 20);
        canvas.present();
        thread::sleep(Duration::from_millis(1));
    }
}

fn cart_pole_feed_forward() {
    let pop_size: usize = 100;
    let input_len: usize = 4;
    let output_len: usize = 1;
    let activation = Activation::Tanh;
    let env = Box::new(CartPoleEnvironment{});
    let mut nn = ANN::new(input_len, output_len, activation);
    nn.add_layer(4, activation);
    let mut pop = Population::new(env, pop_size, nn);

    let mut champion: Option<Genome> = None;
    for g in 0..100 {
        let best = pop.generation();
        println!("gen {} best fitness: {}", g, best.fitness);
        champion = Some(best);
    }

    let genes = champion.unwrap().genes;
    let mut net = ANN::new(input_len, output_len, activation);
    net.add_layer(4, activation);
    net.set_genes(&genes);
    render_champion(net);
}


pub struct Text<'a> {
    context: &'a Sdl2TtfContext,
    filename: &'a str,
    font: Font<'a, 'a>,
    color: Color,
    size: u16,
}

impl<'a> Text<'a> {
    // With color
    pub fn new(ctx: &'a Sdl2TtfContext, filename: &'a str, size: u16, color: Color) -> Result<Text<'a>, String> {
        let font_result = ctx.load_font(Path::new(filename), size);
        if font_result.is_err()  {
            return Err(format!("Failed to initialize font for {}", filename));
        }
        Ok(Text {
            context: ctx,
            filename,
            font: font_result.unwrap(),
            color,
            size,
        })
    }

    pub fn render_surface(&self, text: &'a str) ->  Result<surface::Surface, FontError> {
        let partial = self.font.render(text);
        partial.solid(self.color)
    }

    pub fn set_color(&mut self, color: Color) {
        self.color = color;
    }

    pub fn set_font_size(&mut self, size: u16) {
        let new_font = self.context.load_font(Path::new(self.filename), size).expect("Failed to set font size");
        self.font = new_font;
    }

    pub fn draw(&self, canvas: &mut WindowCanvas, text: &'a str, x: i32, y: i32) {
        let surface = self.render_surface(text).expect("Failed creating surface for font");
        let creator = canvas.texture_creator();
        let texture = creator.create_texture_from_surface(&surface).expect("Failed creating texture");
        let query = texture.query();
        canvas.copy(&texture, None, Rect::new(x, y, query.width, query.height)).expect("Failed copying font texture");
    }

    pub fn draw_multi(&self, canvas: &mut WindowCanvas, text: &'a str, line_spacing: i32, x: i32, y: i32) {
        let split = text.split("[]");
        let mut y_pos = y;
        for line in split {
            let surface = self.render_surface(line).expect("Failed creating surface for font");
            let creator = canvas.texture_creator();
            let texture = creator.create_texture_from_surface(&surface).expect("Failed creating texture");
            let query = texture.query();
            canvas.copy(&texture, None, Rect::new(x, y_pos, query.width, query.height)).expect("Failed copying font texture");
            y_pos += self.size as i32 + line_spacing;
        }
    }
}