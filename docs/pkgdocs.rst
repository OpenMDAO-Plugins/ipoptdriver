
================
Package Metadata
================

- **author:** Herb Schilling

- **author-email:** hschilling@nasa.gov

- **classifier**:: 

    Intended Audience :: Science/Research
    Topic :: Scientific/Engineering

- **description-file:** README.txt

- **entry_points**:: 

    [openmdao.component]
    ipoptdriver.ipoptdriver.IPOPTdriver=ipoptdriver.ipoptdriver:IPOPTdriver
    [openmdao.driver]
    ipoptdriver.ipoptdriver.IPOPTdriver=ipoptdriver.ipoptdriver:IPOPTdriver
    [openmdao.container]
    ipoptdriver.ipoptdriver.IPOPTdriver=ipoptdriver.ipoptdriver:IPOPTdriver

- **home-page:** https://github.com/OpenMDAO-Plugins/ipoptdriver

- **keywords:** openmdao

- **license:** Apache License, Version 2.0

- **maintainer:** Herb Schilling

- **maintainer-email:** hschilling@nasa.gov

- **name:** ipoptdriver

- **project-url:** http://openmdao.org

- **requires-dist:** openmdao.main

- **requires-externals:** Ipopt library from https://projects.coin-or.org/Ipopt and its Python wrapper, Pyipopt from http://code.google.com/p/pyipopt/

- **requires-python**:: 

    >=2.6
    <3.0

- **static_path:** [ '_static' ]

- **summary:** Openmdao driver wrapper for the Ipopt optimization code

- **version:** 0.13

