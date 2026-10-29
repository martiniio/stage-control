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

These licenses cover only this project's own code and content. They grant no rights under any third-party patent or trademark.

- **Code** (all `.py` files): [MIT](LICENSE).
- **Text, figures and videos** (the write-up, generated images and animations): [CC BY 4.0](LICENSE-CONTENT.md). You may share and adapt them, including commercially, as long as you give credit.


## References
1. ASML, US patent US7145269B2, *Lithographic apparatus, Lorentz actuator, and device manufacturing method* — [patents.google.com](https://patents.google.com/patent/US7145269B2/en). Used in 4.1 for the actuator layout: a flat moving coil, two rows of magnets of opposite polarity under its two legs, a steel core closing the flux.

2. M. D. Levenson, "Towards ultimate optical lithography with NXT:1950i dual-stage immersion platform," *Semiconductor Digest* — [sst.semiconductor-digest.com](https://sst.semiconductor-digest.com/?p=30300). Used in 4.1 and 4.2 for the encoder concept (four two-axis heads per stage reading a grid plate) and the sub-nanometre lithography resolution.

3. F. Yang et al., "Two Degree-of-Freedom Fiber-Coupled Heterodyne Grating Interferometer with Milli-Radian Operating Range of Rotation," *Sensors* 19(14), 3219, 2019 — [mdpi.com](https://www.mdpi.com/1424-8220/19/14/3219). Used in 4.2: it states that lithography encoder heads must work over ±1.5 mrad of rotation.

4. HEIDENHAIN PP 281 two-coordinate encoder, product data — [micromech.co.uk](https://micromech.co.uk/Product/101072/pm), [heidenhain.us](https://www.heidenhain.us/?p=493). Used in 4.2 for the catalogue encoder step (down to 0.05 µm).
5. Renishaw TONiC readhead installation drawing — [pdf](https://8id.xray.aps.anl.gov/elog-mirror/8-ID/170816_110154/TONiC.pdf). Used in 4.2 for how closely a linear readhead must follow its scale.

6. T. Flash and N. Hogan, "The coordination of arm movements: an experimentally confirmed mathematical model," *Journal of Neuroscience* 5(7), 1688–1703, 1985 — [jneurosci.org](https://www.jneurosci.org/content/5/7/1688). Used in 7.4 for the minimum-jerk curve.