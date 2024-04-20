# CiMLoop

Welcome to the CiMLoop repository! CiMLoop (Andrulis, ISPASS 2024) is a
full-stack CiM modeling tool with flexible user-defined systems and fast,
accurate statistical energy modeling.

This repository contains tutorials, examples, documentation, and an artifact for
the CiMLoop paper. All are accessible through the Docker container.

**Note: This repository is a work in progress. More tutorials and documentation
are on the way!**

## Quick Start

```bash
git clone https://github.com/mit-emze/cimloop.git
cd cimloop
export DOCKER_ARCH=<your processor architecture. supported: amd64, arm64>
docker-compose pull
docker-compose up
# Connect to the container and explore CiMLoop! The README.md file
# in the workspace directory contains more information.

# If you have permission issues, please see the instructions in the
# docker-compose.yaml file on how to set the UID and GID.
```

The [README.md in the workspace](workspace/README.md) directory contains more
information on how to use CiMLoop.

## Tutorials
The [Timeloop and Accelergy tutorials]
(https://github.com/Accelergy-Project/timeloop-accelergy-exercises) are a
prerequisite for using CiMLoop as well, so please complete those first. CiMLoop
tutorials are available in the `workspace/tutorials` directory.

## CiMLoop Artifact
CiMLoop published results can be reproduced by running the
`models/arch/1_macro/*/_guide.ipynb` notebooks and the
`tutorials/demo_speed_accuracy.ipynb` notebook in the workspace.

## CiMLoop for Photonic Accelerators
CiMLoop includes a model of the [Albireo silicon photonic
accelerator](workspace/models/arch/1_macro/albireo_isca_2021) as described in
"Architecture-Level Modeling of Photonic Deep Neural Network Accelerators"
(Andrulis et al., ISPASS 2024).

## Documentation
Documentation can be found at the following locations:

- [CiMLoop Paper (Not yet available)]()
- [Timeloop+Accelergy Documentation](https://timeloop.csail.mit.edu/)
- [Timeloop+Accelergy Exercises](https://github.com/Accelergy-Project/timeloop-accelergy-exercises)
- [Timeloop Front-End Documentation](https://accelergy-project.github.io/timeloopfe/)

## Contributing
If you would like to contribute models of memory cells, components,
architectures, workloads, or anything else, please submit a pull request. We
welcome contributions!

## Citation
If you use CiMLoop in your research, please cite the papers in
[citations.bib](citations.bib).

