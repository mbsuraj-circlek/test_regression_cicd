# Databricks notebook source
import pandas as pd
import json
from tempfile import TemporaryDirectory
from pandas.testing import assert_frame_equal

# COMMAND ----------

class Regression_tests:
  def __init__(self, notebook_loc):
    self.notebook_loc = notebook_loc
    self.count = 0
    self.total_tests = 2
    
  def test_input_parameters(self):
    print("1. Test Input Parameters->", end=" ")
    assert self.notebook_loc!="", "INPUT ERROR: No Valid location provided."
    print("Success")
    self.count += 1
    
  def test_regression_model(self):
    print("\n2. Test Regression Model->", end=" ")
    result = json.loads(dbutils.notebook.run(self.notebook_loc, 0))
#     result = dbutils.notebook.run(self.notebook_loc)
    threshold_r2 = 0.8
    assert result['r2'] > 0.80, f"TEST FAILED: The model doesn't explain variance in the data sufficiently. Threshold r2: {threshold_r2}, Model_r2: {result['r2']}"
    print(result, "Success")
    self.count += 1

  
  def run_all_tests(self):
    try:
      self.test_input_parameters()
      self.test_regression_model()
      print("All tests are successful! ")
    except Exception as e:
      print(e)
      print(f"Total Successful tests: {self.count}")

# COMMAND ----------

test_object = Regression_tests("../test_regression_cicd/regression")
test_object.run_all_tests()

# COMMAND ----------

test_object2 = Regression_tests("")
test_object2.run_all_tests()

# COMMAND ----------


