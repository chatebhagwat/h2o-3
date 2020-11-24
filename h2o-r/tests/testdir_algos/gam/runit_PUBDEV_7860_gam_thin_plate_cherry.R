setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../../scripts/h2o-r-test-setup.R")
library(mgcv)
library(tidyverse)


# simple test to make sure we can invoke early stop in R for GAM.  Heavy testing is done in Python.  We
# are just testing the client API.
test.model.gam.thin.plates <- function() {
    h2oTrain <- h2o.importFile("/Users/wendycwong/temp/Gaussian80Rows.csv")
    rTrain <- as.data.frame(h2oTrain)
    data(trees)
    browser()
    ct2 <- gam(C21 ~ s(C17, C18, C19, C20, k=7, sp = 0.1, bs = "tp"), family=Gamma(link=log), data=rTrain)
   # ct2 <- gam(C21 ~ s(C18, C19, C20, k=7, sp = 0.1)+s(C17, C18, k=6, sp=0.1), family=Gamma(link=log), data=rTrain)
  #  ct1 <- gam(Volume ~ s(Height, Girth, k = 6, sp = 0.1), family=Gamma(link=log), data=trees)

}

doTest("GAM with thin plate regression splines with cherry", test.model.gam.thin.plates)
