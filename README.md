# Planar Stage Control

A simplified 3-DOF planar **XY-θz precision stage**, inspired by the in-plane motion problem of wafer stages, modelled from first principles and controlled in simulation.

The plate floats on air bearings, is pushed by four voice coils and is measured by four grid-encoder heads. Everything is built up from geometry, magnets and Newton: the coil force constant comes from a 2-D field solve, and the equations of motion from the coil layout. Then five control experiments show what feedback, a planned path and model feedforward each add.

The full story is in the blog post: **[Stage Control: Modelling a Planar Precision Stage from First Principles](https://my-website-kohl-eta-22.vercel.app/blog/)**.

## Run it

```bash
pip install -r requirements.txt
python stage_control.py      # experiments 1–5 → pictures/ and data/   (about 1 minute)
python stage_animation.py    # videos from data/ → pictures/video/     (about 10 minutes)
```

The animations also need [ffmpeg](https://ffmpeg.org/) on your PATH. To check one drawing quickly without making a video: `python stage_animation.py 5 --still`.

## What is in here

| File | What it does |
|---|---|
| `stage_model.py` | The plant: parameters, the input matrix $B(\theta_z)$ and the nonlinear model as a [python-control](https://python-control.readthedocs.io/) system |
| `control.py` | The controller (PIDF, minimum-jerk paths, feedforward, coil allocation) and the five experiments; saves figures and all logged signals |
| `stage_animation.py` | Top-view videos of the experiments, drawn from the saved data |
| `sheets/plant_sheet.py` | Drawing sheets 1a and 1b: top view and section of the stage |
| `sheets/actuator_sheet.py` | Drawing sheets 2a and 2b: voice-coil field and force constant |
| `sheets/vc_field_solver.py` | The 2-D magnetic field solve of one voice coil |
| `sheets/plant_sim.py` | Nine physics checks of the plant model |

## The experiments

1. **Step + PIDF:** feedback alone gets there, but asks for a large, abrupt current.
2. **Minimum-jerk path + PIDF:** a planned path makes the request reasonable.
3. **Path + PIDF + feedforward:** the model supplies the force; the error drops below a nanometre.
4. **Spiral:** a sequence of targets joined into one master path.
5. **Raster scan:** a grid over a 4 × 2 mm area, starting from a turned plate.

## Disclaimer

This is a personal, educational project. It is not affiliated with, endorsed by or based on any information from ASML or any other company. Everything comes from public sources (cited in the blog post), from textbook physics, or from assumptions stated where they are made. The model is deliberately simplified — rigid plate, ideal amplifiers and sensors, no heat — and is not a design for a real stage.

## License

- **Code** (all `.py` files): [MIT](LICENSE).
- **Text, figures and videos** (the write-up, generated images and animations): [CC BY 4.0](LICENSE-CONTENT.md). You may share and adapt them, including commercially, as long as you give credit.