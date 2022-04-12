# `vishelper` 
Wrapper functions for easy, nicely formatted visuals for exploratory data analysis. 


## Installation
To install the package, do the following: 

1. Clone this repository: 

  ```bash
  git clone git@github.com:cmawer/vishelper.git
  ``` 

2. Install the repository as a Python package. From the root of this repo: 

  ```bash
  pip install .
  ```

## Dependencies
Install the packages in `requirements.txt`  

```bash
pip install -r requirements.txt 
```

The functions in this package have been built to be fairly flexible with various versions of the packages required but the enclosed versions are those that have been used.

Note: `selenium` is only required to run `vishelper.save_map()` with `png=True` or `vishelper.html_to_png()`. If not installed, the package can still be imported and the code will only break if either of these functions is executed. 

## Guides and documentation 
See notebooks in `demos/` for step-by-step guides. 

See documentation [here](https://cmawer.github.io/vishelper/).
