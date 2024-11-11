# ilumr-courseware

Lab courses for [ilumr: the Educational MRI System by Resonint](https://resonint.com).

## Installation on ilumr

Note: If you are still using ilumr software version 2.1.0, you will need to update the ilumr to at least version 2.2.0 to include the Experiment library that is used in the latest version of the courseware. Alternately you can use the v0.1.0 release of this repository with ilumr v2.1.0.

#### Method 1: With `git`

Requires internet access on the ilumr system via its ethernet connection.

Open a terminal in the [jupyterlab interface](https://jupyterlab.readthedocs.io/en/stable/user/terminal.html) and run:

    git clone https://github.com/Resonint/ilumr-courseware.git

#### Method 2: By downloading a zip file

1. Download the repository as a zip file
2. Drag the zip file from your downloads folder into the [jupyterlab file browser](https://jupyterlab.readthedocs.io/en/stable/user/files.html#uploading-and-downloading)
3. Open a terminal in the [jupyterlab interface](https://jupyterlab.readthedocs.io/en/stable/user/terminal.html) and run:
    
        unzip ilumr-courseware-main.zip

## Usage

The notebooks under the `courses` directory can be opened using the jupyterlab file browser and executed.

Note: The notebooks load python modules from the `dashboards-inline` directory using relative imports, so they must be run from the directory structure of this repository. To create a copy for e.g. students to use it is recommended to either use the `git clone` or `unzip` methods above.

The notebooks can also be viewed on a PC with jupyterlab installed (but generally not executed, as the `matipo` library is only available on ilumr).

## Status

### MRI Fundamentals

| #   | Topic                 | Status         |
| --- | --------------------- | -------------- |
| 1   | Intro to NMR          | Complete       |
| 2   | Imaging & K-Space     | Complete       |
| 3   | Selective Excitation  | Complete       |
| 4   | Relaxation & Contrast | Second Draft   |
| 5   | Fast Imaging          | Planned        |
| 6   | Imaging Artefacts     | Planned        |
| 7   | Flow & Diffusion      | Planned        |
| 8   | Non-Cartesian Imaging | Planned        |

## Contributing

Pull requests are welcome. For major changes and feature requests, please visit the [Resonint public forum](https://resonint.discourse.group/c/ilumr-courseware/) to discuss what you would like to change.

Issues are appreciated for reporting bugs. For general support, create a topic in the forum's [support category](https://resonint.discourse.group/c/support/).

## License

[MIT](LICENSE)

## Acknowledgments

This project has contributions from the following people:
- Cameron Dykstra
- Sharon McTaggart
- Sergei Obruchkov
- Cam Nowikow
- Mike Noseworthy
