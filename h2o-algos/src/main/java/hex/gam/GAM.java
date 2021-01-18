package hex.gam;

import hex.DataInfo;
import hex.ModelBuilder;
import hex.ModelCategory;
import hex.ModelMetrics;
import hex.gam.GAMModel.GAMParameters;
import hex.gam.MatrixFrameUtils.GamUtils;
import hex.gam.MatrixFrameUtils.GenerateGamMatrixOneColumn;
import hex.glm.GLM;
import hex.glm.GLMModel;
import hex.glm.GLMModel.GLMParameters;
import jsr166y.ForkJoinTask;
import jsr166y.RecursiveAction;
import water.*;
import water.exceptions.H2OModelBuilderIllegalArgumentException;
import water.fvec.Frame;
import water.fvec.Vec;
import water.util.ArrayUtils;
import water.util.IcedHashSet;
import water.util.Log;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static hex.gam.GAMModel.cleanUpInputFrame;
import static hex.gam.MatrixFrameUtils.GAMModelUtils.copyGLMCoeffs;
import static hex.gam.MatrixFrameUtils.GAMModelUtils.copyGLMtoGAMModel;
import static hex.gam.MatrixFrameUtils.GamUtils.*;
import static hex.gam.MatrixFrameUtils.GamUtils.AllocateType.*;
import static hex.genmodel.utils.ArrayUtils.flat;
import static hex.glm.GLMModel.GLMParameters.Family.multinomial;
import static hex.glm.GLMModel.GLMParameters.Family.ordinal;
import static hex.glm.GLMModel.GLMParameters.GLMType.gam;
import static water.util.ArrayUtils.longRandomVector;
import static water.util.ArrayUtils.totalArrayDimension;


public class GAM extends ModelBuilder<GAMModel, GAMModel.GAMParameters, GAMModel.GAMModelOutput> {
  private double[][][] _knots; // Knots for splines
  private double[] _cv_alpha = null;  // best alpha value found from cross-validation
  private double[] _cv_lambda = null; // bset lambda value found from cross-validation
  private boolean _thin_plate_smoothers_used = false;
  
  @Override
  public ModelCategory[] can_build() {
    return new ModelCategory[]{ModelCategory.Regression};
  }

  @Override
  public boolean isSupervised() {
    return true;
  }

  @Override
  public BuilderVisibility builderVisibility() {
    return BuilderVisibility.Experimental;
  }

  @Override
  public boolean havePojo() {
    return false;
  }

  @Override
  public boolean haveMojo() {
    return true;
  }

  public GAM(boolean startup_once) {
    super(new GAMModel.GAMParameters(), startup_once);
  }

  public GAM(GAMModel.GAMParameters parms) {
    super(parms);
    init(false);
  }

  public GAM(GAMModel.GAMParameters parms, Key<GAMModel> key) {
    super(parms, key);
    init(false);
  }

  // cross validation can be used to choose the best alpha/lambda values among a whole collection of alpha
  // and lambda values.  Future hyperparameters can be added for cross-validation to choose as well.
  @Override
  public void computeCrossValidation() {
    if (error_count() > 0) {
      throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(GAM.this);
    }
    super.computeCrossValidation();
  }

  // find the best alpha/lambda values used to build the main model moving forward by looking at the devianceValid
  @Override
  public void cv_computeAndSetOptimalParameters(ModelBuilder[] cvModelBuilders) {
    double deviance_valid = Double.POSITIVE_INFINITY;
    double best_alpha = 0;
    double best_lambda = 0;
    for (int i = 0; i < cvModelBuilders.length; ++i) {  // run cv for each lambda value
      GAMModel g = (GAMModel) cvModelBuilders[i].dest().get();
      if (g._output._devianceValid < deviance_valid) {
        best_alpha= g._output._best_alpha;
        best_lambda = g._output._best_lambda;
      }
    }
    _cv_alpha = new double[]{best_alpha};
    _cv_lambda = new double[]{best_lambda};
  }
    /***
     * This method will look at the keys of knots stored in _parms._knot_ids and copy them over to double[][]
     * array.  Note that we have smoothers that take different number of columns and we will keep the order of 
     * the knots as specified by the user in _parm._gam_columns.
     *
     * @return double[][] array containing the knots specified by users
     */
  public double[][][] generateKnotsFromKeys() { // todo: parallize this operation
    int numGamCols = totalArrayDimension(_parms._gam_columns); // total number of predictors in all smoothers
    double[][][] knots = new double[numGamCols][][]; // 1st index into gam column, 2nd index number of knots for the row
    boolean allNull = _parms._knot_ids == null;
    for (int outIndex = 0; outIndex < _parms._gam_columns.length; outIndex++) {
      String tempKey = allNull ? null : _parms._knot_ids[outIndex]; // one knot_id for each smoother
      knots[outIndex] = new double[_parms._gam_columns[outIndex].length][];
      if (tempKey != null && (tempKey.length() > 0)) {  // read knots location from Frame given by user      
        final Frame knotFrame = Scope.track((Frame) DKV.getGet(Key.make(tempKey)));
        double[][] knotContent = new double[(int) knotFrame.numRows()][_parms._gam_columns[outIndex].length];
        final ArrayUtils.FrameToArray f2a = new ArrayUtils.FrameToArray(0,
                _parms._gam_columns[outIndex].length-1, knotFrame.numRows(), knotContent);
        knotContent = f2a.doAll(knotFrame).getArray();  // first index is row, second index is column
       
        final double[][] knotCTranspose = ArrayUtils.transpose(knotContent);// change knots to correct order
        for (int innerIndex = 0; innerIndex < knotCTranspose.length; innerIndex++) {
          knots[outIndex][innerIndex] = new double[knotContent.length];
          System.arraycopy(knotCTranspose[innerIndex], 0, knots[outIndex][innerIndex], 0, 
                  knots[outIndex][innerIndex].length);
          if (knotCTranspose.length == 1 && _parms._bs[outIndex] == 0) // only check for order to single smoothers
            failVerifyKnots(knots[outIndex][innerIndex], outIndex);
        }
        _parms._num_knots[outIndex] = knotContent.length;

      } else {  // current column knotkey is null, we will use default method to generate knots
        final Frame predictVec = new Frame(_parms._gam_columns[outIndex],
                _parms.train().vecs(_parms._gam_columns[outIndex]));
        if (_parms._gam_columns[outIndex].length == 1) {
          knots[outIndex][0] = generateKnotsOneColumn(predictVec, _parms._num_knots[outIndex]);
          failVerifyKnots(knots[outIndex][0], outIndex);
        } else {  // generate knots for multi-predictor smooths, randomly choose rows in parms._num_knots
          long[] randomRowVec = longRandomVector(_parms._seed, 
                  _parms._num_knots[outIndex]+(int)predictVec.naCount(), predictVec.numRows());
          int rowCount = 0;
          knots[outIndex] = MemoryManager.malloc8d(_parms._gam_columns[outIndex].length, _parms._num_knots[outIndex]);
          boolean foundNAinRow = false;
          for (int rowInd = 0; rowInd < randomRowVec.length; rowInd++) {
            if (rowCount >= _parms._num_knots[outIndex])
              break;
            for (int colInd = 0; colInd < _parms._gam_columns[outIndex].length; colInd++) {
              foundNAinRow = false;
              if (Double.isNaN((predictVec.vec(_parms._gam_columns[outIndex][colInd]).at(randomRowVec[rowInd])))) {
                foundNAinRow = true;
                break;  // check for na
              }
              knots[outIndex][colInd][rowCount] = predictVec.vec(_parms._gam_columns[outIndex][colInd]).at(randomRowVec[rowInd]);
            }
            if (!foundNAinRow)
              rowCount++;
          }
        }
      }
    }
    return knots;
  }
  
  // this function will check and make sure the knots location specified in knots are valid in the following sense:
  // 1. They do not contain NaN
  // 2. They are sorted in ascending order.
  public void failVerifyKnots(double[] knots, int gam_column_index) {
    for (int index = 0; index < knots.length; index++) {
      if (Double.isNaN(knots[index])) {
        error("gam_columns/knots_id", String.format("Knots generated by default or specified in knots_id " +
                        "ended up containing a NaN value for gam_column %s.   Please specify alternate knots_id" +
                        " or choose other columns.", _parms._gam_columns[gam_column_index][0]));
        return;
      }
      if (index > 0 && knots[index - 1] > knots[index]) {
        error("knots_id", String.format("knots not sorted in ascending order for gam_column %s. " +
                        "Knots at index %d: %f.  Knots at index %d: %f",_parms._gam_columns[gam_column_index][0], index-1, 
                knots[index-1], index, knots[index]));
        return;
      }
      if (index > 0 && knots[index - 1] == knots[index]) {
        error("gam_columns/knots_id", String.format("chosen gam_column %s does have not enough values to " +
                        "generate well-defined knots. Please choose other columns or reduce " +
                        "the number of knots.  If knots are specified in knots_id, choose alternate knots_id as the" +
                        " knots are not in ascending order.  Knots at index %d: %f.  Knots at index %d: %f", 
                _parms._gam_columns[gam_column_index][0], index-1, knots[index-1], index, knots[index]));
        return;
      }
    }
  }
  
  @Override
  public void init(boolean expensive) {
    super.init(expensive);
    if (expensive)  // add GAM specific check here
      validateGamParameters();
  }
  
  private void validateGamParameters() {
    if (_parms._max_iterations == 0)
      error("_max_iterations", H2O.technote(2, "if specified, must be >= 1."));
    if (_parms._family == GLMParameters.Family.AUTO) {
      if (nclasses() == 1 & _parms._link != GLMParameters.Link.family_default && _parms._link != GLMParameters.Link.identity
              && _parms._link != GLMParameters.Link.log && _parms._link != GLMParameters.Link.inverse && _parms._link != null) {
        error("_family", H2O.technote(2, "AUTO for undelying response requires the link to be family_default, identity, log or inverse."));
      } else if (nclasses() == 2 & _parms._link != GLMParameters.Link.family_default && _parms._link != GLMParameters.Link.logit
              && _parms._link != null) {
        error("_family", H2O.technote(2, "AUTO for undelying response requires the link to be family_default or logit."));
      } else if (nclasses() > 2 & _parms._link != GLMParameters.Link.family_default & _parms._link != GLMParameters.Link.multinomial
              && _parms._link != null) {
        error("_family", H2O.technote(2, "AUTO for undelying response requires the link to be family_default or multinomial."));
      }
    }
    if (error_count() > 0)
      throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(GAM.this);
    if (_parms._gam_columns == null)  // check _gam_columns contains valid columns
      error("_gam_columns", "must specify columns names to apply GAM to.  If you don't have any," +
              " use GLM.");
    else  // check and make sure gam_columns column types are legal
      assertLegalGamColumnsNBSTypes();
    if (_parms._bs == null) {
      _thin_plate_smoothers_used = _thin_plate_smoothers_used || setDefaultBSType(_parms);
    }
    if ((_parms._bs != null) && (_parms._gam_columns.length != _parms._bs.length))  // check length
      error("gam colum number", "Number of gam columns implied from _bs and _gam_columns do not " +
              "match.");
    if (_thin_plate_smoothers_used)
      setThinPlateParameters(_parms); // set the m, M for thin plate regression smoothers
    checkOrChooseNumKnots(); // check valid num_knot assignment or choose num_knots
    if ((_parms._num_knots != null) && (_parms._num_knots.length != _parms._gam_columns.length))
      error("gam colum number", "Number of gam columns implied from _num_knots and _gam_columns do" +
              " not match.");
    if (_parms._knot_ids != null) { // check knots location specification
      if (_parms._knot_ids.length != _parms._gam_columns.length)
        error("gam colum number", "Number of gam columns implied from _num_knots and _knot_ids do" +
                " not match.");
    }
    _knots = generateKnotsFromKeys(); // generate knots and verify that they are given correctly
    checkThinPlateParams();
    if (_parms._saveZMatrix && ((_train.numCols() - 1 + _parms._num_knots.length) < 2))
      error("_saveZMatrix", "can only be enabled if we number of predictors plus" +
              " Gam columns in gam_columns exceeds 2");
    if ((_parms._lambda_search || !_parms._intercept || _parms._lambda == null || _parms._lambda[0] > 0))
      _parms._use_all_factor_levels = true;
    if (_parms._link == null) {
      _parms._link = GLMParameters.Link.family_default;
    }
    if (_parms._family == GLMParameters.Family.AUTO) {
      if (_nclass == 1) {
        _parms._family = GLMParameters.Family.gaussian;
      } else if (_nclass == 2) {
        _parms._family = GLMParameters.Family.binomial;
      } else {
        _parms._family = GLMParameters.Family.multinomial;
      }
    }
    if (_parms._link == null || _parms._link.equals(GLMParameters.Link.family_default))
      _parms._link = _parms._family.defaultLink;
    
    if ((_parms._family == GLMParameters.Family.multinomial || _parms._family == GLMParameters.Family.ordinal ||
            _parms._family == GLMParameters.Family.binomial)
            && response().get_type() != Vec.T_CAT) {
      error("_response_column", String.format("For given response family '%s', please provide a categorical" +
              " response column. Current response column type is '%s'.", _parms._family, response().get_type_str()));
    }
  }
  
  public void checkThinPlateParams() {
    if (!_thin_plate_smoothers_used)
      return;
    
    int numGamCols = _parms._gam_columns.length;
    for (int index = 0; index < numGamCols; index++) {
      if (_parms._bs[index] == 1) {
        if (_parms._num_knots[index] <= _parms._M[index]+1)
          error("num_knots", "num_knots for gam column start with  "+_parms._gam_columns[index][0]+
                  " did not specify enough num_knots.  It should be "+(_parms._M[index]+1)+" or higher.");
      }
    }
  }
  
  // set default num_knots to 10 for gam_columns where there is no knot_id specified
  public void checkOrChooseNumKnots() {
    if (_parms._num_knots == null)
      _parms._num_knots = new int[_parms._gam_columns.length];  // different columns may have different
    for (int index = 0; index < _parms._num_knots.length; index++) {  // set zero value _num_knots
      if (_parms._knot_ids == null || (_parms._knot_ids != null && _parms._knot_ids[index] == null)) {  // knots are not specified
        int numKnots = _parms._num_knots[index];
        int naSum = 0;
        for (int innerIndex = 0; innerIndex < _parms._gam_columns[index].length; innerIndex++) {
          naSum += _parms.train().vec(_parms._gam_columns[index][innerIndex]).naCnt();
        }
        long eligibleRows = _train.numRows()-naSum;
        if (_parms._num_knots[index] == 0) {  // set num_knots to default
          int defaultRows = 10;
          if (_parms._bs[index] == 1)
            defaultRows = Math.max(defaultRows, _parms._M[index]+1);
          _parms._num_knots[index] = eligibleRows < defaultRows ? (int) eligibleRows : defaultRows;
        } else {  // num_knots assigned by user and check to make sure it is legal
          if (numKnots > eligibleRows) {
            error("_num_knots", " number of knots specified in _num_knots: "+numKnots+" for smoother" +
                    " with first predictor "+_parms._gam_columns[index][0]+".  Reduce _num_knots.");
          }
        }
      }
    }
  }
  
  public void assertLegalGamColumnsNBSTypes() {
    Frame dataset = _parms.train();
    List<String> cNames = Arrays.asList(dataset.names());
    for (int index = 0; index < _parms._gam_columns.length; index++) {
      if (_parms._bs != null) { // check and make sure the correct bs type is chosen
        if (_parms._bs[index] == 1) // todo add support for bs==2
          _thin_plate_smoothers_used = true;
        if (_parms._gam_columns[index].length == 1 && _parms._bs[index] != 0) 
          error("bs", "column name" + _parms._gam_columns[index][0]+" is the single predictor of" +
                  " a smoother and can only use bs = 0");
        else if (_parms._gam_columns[index].length > 1 && _parms._bs[index] != 1)
          error("bs", "Smother with multiple predictors can only use bs = 1");        
      }
        
      for (int innerIndex = 0; innerIndex < _parms._gam_columns[index].length; innerIndex++) {
        String cname = _parms._gam_columns[index][innerIndex];
        if (!cNames.contains(cname))
          error("gam_columns", "column name: " + cname + " does not exist in your dataset.");
        if (dataset.vec(cname).isCategorical())
          error("gam_columns", "column " + cname + " is categorical and cannot be used as a gam " +
                  "column.");
        if (dataset.vec(cname).isBad() || dataset.vec(cname).isTime() || dataset.vec(cname).isUUID() ||
                dataset.vec(cname).isConst())
          error("gam_columns", String.format("Column '%s' of type '%s' cannot be used as GAM column. Column types " +
                  "BAD, TIME, CONSTANT and UUID cannot be used.", cname, dataset.vec(cname).get_type_str()));
        if (!dataset.vec(cname).isNumeric())
          error("gam_columns", "column " + cname + " is not numerical and cannot be used as a gam" +
                  " column.");
      }
    }
  }

  @Override
  protected boolean computePriorClassDistribution() {
    return (_parms._family== multinomial)||(_parms._family== ordinal);
  }

  @Override
  protected GAMDriver trainModelImpl() {
    return new GAMDriver();
  }

  @Override
  protected int nModelsInParallel(int folds) {
    return nModelsInParallel(folds,2);
  }

  private class GAMDriver extends Driver {
    double[][][] _zTranspose; // store for each GAM predictor transpose(Z) matrix
    double[][][] _penalty_mat_center;  // store for each GAM predictor the penalty matrix
    double[][][] _penalty_mat;  // penalty matrix before centering
    public double[][][] _binvD; // store BinvD for each gam column specified for scoring
    public int[] _numKnots;  // store number of knots per gam column
    String[][] _gamColNames;  // store column names of GAM columns
    String[][] _gamColNamesCenter;  // gamColNames after de-centering is performed.
    Key<Frame>[] _gamFrameKeys;
    Key<Frame>[] _gamFrameKeysCenter;
    double[][] _gamColMeans; // store gam column means without centering.
    /***
     * This method will take the _train that contains the predictor columns and response columns only and add to it
     * the following:
     * 1. For each predictor included in gam_columns, expand it out to calculate the f(x) and attach to the frame.
     * 2. It will calculate the ztranspose that is used to center the gam columns.
     * 3. It will calculate a penalty matrix used to control the smoothness of GAM.
     *
     * @return
     */
    Frame adaptTrain() {
      int numGamFrame = _parms._gam_columns.length;
      _zTranspose = GamUtils.allocate3DArray(numGamFrame, _parms, firstOneLess);
      _penalty_mat = _parms._savePenaltyMat?GamUtils.allocate3DArray(numGamFrame, _parms, sameOrig):null;
      _penalty_mat_center = GamUtils.allocate3DArray(numGamFrame, _parms, bothOneLess);
      _binvD = GamUtils.allocate3DArray(numGamFrame, _parms, firstTwoLess);
      _numKnots = MemoryManager.malloc4(numGamFrame);
      _gamColNames = new String[numGamFrame][];
      _gamColNamesCenter = new String[numGamFrame][];
      _gamFrameKeys = new Key[numGamFrame];
      _gamFrameKeysCenter = new Key[numGamFrame];
      _gamColMeans = new double[numGamFrame][];

      addGAM2Train();  // add GAM columns to training frame
      return buildGamFrame(_parms, _train, _gamFrameKeysCenter); // add gam cols to _train
    }
    
    public class ThinPlateRegressionSmootherWithKnots extends RecursiveAction {

      @Override
      protected void compute() {
        
      }
    }
    
    public class CubicSplineSmoother extends RecursiveAction {
      final Frame _predictVec;
      final int _numKnots;
      final int _numKnotsM1;
      final int _splineType;
      final boolean _savePenaltyMat;
      final String[] _newColNames;
      final double[] _knots;
      final GAMParameters _parms;
      final AllocateType _fileMode;
      final int _gamColIndex;
      
      public CubicSplineSmoother(Frame predV, GAMParameters parms, int gamColIndex, String[] gamColNames, double[] knots,
                                 AllocateType fileM) {
        _predictVec = predV;
        _numKnots = parms._num_knots[gamColIndex];
        _numKnotsM1 = _numKnots-1;
        _splineType = parms._bs[gamColIndex];
        _savePenaltyMat = parms._savePenaltyMat;
        _newColNames = gamColNames;
        _knots = knots;
        _parms = parms;
        _gamColIndex = gamColIndex;
        _fileMode = fileM;
      }

      @Override
      protected void compute() {
        GenerateGamMatrixOneColumn genOneGamCol = new GenerateGamMatrixOneColumn(_splineType, _numKnots,
                _knots, _predictVec).doAll(_numKnots, Vec.T_NUM, _predictVec);
        if (_savePenaltyMat)  // only save this for debugging
          GamUtils.copy2DArray(genOneGamCol._penaltyMat, _penalty_mat[_gamColIndex]); // copy penalty matrix
        Frame oneAugmentedColumnCenter = genOneGamCol.outputFrame(Key.make(), _newColNames,
                null);
        oneAugmentedColumnCenter = genOneGamCol.centralizeFrame(oneAugmentedColumnCenter,
                _predictVec.name(0) + "_" + _splineType + "_center_", _parms);
        GamUtils.copy2DArray(genOneGamCol._ZTransp, _zTranspose[_gamColIndex]); // copy transpose(Z)
        double[][] transformedPenalty = ArrayUtils.multArrArr(ArrayUtils.multArrArr(genOneGamCol._ZTransp,
                genOneGamCol._penaltyMat), ArrayUtils.transpose(genOneGamCol._ZTransp));  // transform penalty as zt*S*z
        GamUtils.copy2DArray(transformedPenalty, _penalty_mat_center[_gamColIndex]);
        _gamFrameKeysCenter[_gamColIndex] = oneAugmentedColumnCenter._key;
        DKV.put(oneAugmentedColumnCenter);
        System.arraycopy(oneAugmentedColumnCenter.names(), 0, _gamColNamesCenter[_gamColIndex], 0,
                _numKnotsM1);
        GamUtils.copy2DArray(genOneGamCol._bInvD, _binvD[_gamColIndex]);
      }
    }

    void addGAM2Train() {
      int numGamFrame = _parms._gam_columns.length; // number of smoothers to generate
      RecursiveAction[] generateGamColumn = new RecursiveAction[numGamFrame];
      for (int index = 0; index < numGamFrame; index++) { // generate smoothers/splines
        final Frame predictVec = prepareGamVec(index, _parms);  // extract predictors from training frame
        _gamColNames[index] = generateGamColNames(index, _parms);
        int numKnots = _parms._num_knots[index];
        int numKnotsM1 = numKnots - 1;
        if (_parms._gam_columns[index].length == 1 && _parms._bs[index] == 0) {// single predictor smoothers
          _gamColNamesCenter[index] = new String[numKnotsM1];
          _gamColMeans[index] = new double[numKnots];
          generateGamColumn[index] = new CubicSplineSmoother(predictVec, _parms, index, _gamColNames[index],
                  _knots[index][0], firstTwoLess);
        } else if (_parms._bs[index] == 1) {
          generateGamColumn[index] = new ThinPlateRegressionSmootherWithKnots();
        }
      }
      ForkJoinTask.invokeAll(generateGamColumn);
    }

    void verifyGamTransformedFrame(Frame gamTransformed) {
      int numGamCols = _gamColNamesCenter.length;
      int numGamFrame = _parms._gam_columns.length;
      for (int findex = 0; findex < numGamFrame; findex++) {
        for (int index = 0; index < numGamCols; index++) {
          if (gamTransformed.vec(_gamColNamesCenter[findex][index]).isConst())
            error(_gamColNamesCenter[findex][index], "gam column transformation generated constant columns" +
                    " for " + _parms._gam_columns[findex]);
        }
      }
    }
    
    @Override
    public void computeImpl() {
      init(true);
      if (error_count() > 0)   // if something goes wrong, let's throw a fit
        throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(GAM.this);
      Frame newTFrame = new Frame(rebalance(adaptTrain(), false, _result+".temporary.train"));  // get frames with correct predictors and spline functions
      verifyGamTransformedFrame(newTFrame);
      
      if (error_count() > 0)   // if something goes wrong during gam transformation, let's throw a fit again!
        throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(GAM.this);
      
      if (valid() != null) {  // transform the validation frame if present
        _valid = rebalance(cleanUpInputFrame(_parms.valid().clone(), _parms, _gamColNamesCenter, _binvD, _zTranspose, _knots,
                _numKnots), false, _result+".temporary.valid");
      }
      DKV.put(newTFrame); // This one will cause deleted vectors if add to Scope.track
      Frame newValidFrame = _valid == null ? null : new Frame(_valid);
      if (newValidFrame != null) {
        DKV.put(newValidFrame);
      }

      _job.update(0, "Initializing model training");
      buildModel(newTFrame, newValidFrame); // build gam model

    }


    public final void buildModel(Frame newTFrame, Frame newValidFrame) {
      GAMModel model = null;
      DataInfo dinfo = null;
      final IcedHashSet<Key<Frame>> validKeys = new IcedHashSet<>();
      try {
        _job.update(0, "Adding GAM columns to training dataset...");
        dinfo = new DataInfo(_train.clone(), _valid, 1, _parms._use_all_factor_levels 
                || _parms._lambda_search, _parms._standardize ? 
                DataInfo.TransformType.STANDARDIZE : DataInfo.TransformType.NONE, DataInfo.TransformType.NONE,
                _parms.missingValuesHandling() == GLMParameters.MissingValuesHandling.Skip,
                _parms.missingValuesHandling() == GLMParameters.MissingValuesHandling.MeanImputation 
                        || _parms.missingValuesHandling() == GLMParameters.MissingValuesHandling.PlugValues,
                _parms.makeImputer(), false, hasWeightCol(), hasOffsetCol(), hasFoldCol(),
                _parms.interactionSpec());
        DKV.put(dinfo._key, dinfo);
        model = new GAMModel(dest(), _parms, new GAMModel.GAMModelOutput(GAM.this, dinfo));
        model.write_lock(_job);
        if (_parms._keep_gam_cols) {  // save gam column keys
          model._output._gamTransformedTrainCenter = newTFrame._key;
        }
        _job.update(1, "calling GLM to build GAM model...");
        GLMModel glmModel = buildGLMModel(_parms, newTFrame, newValidFrame); // obtained GLM model
        if (model.evalAutoParamsEnabled) {
          model.initActualParamValuesAfterGlmCreation();
        }
        Scope.track_generic(glmModel);
        _job.update(0, "Building out GAM model...");
        fillOutGAMModel(glmModel, model); // build up GAM model
        model.update(_job);
        // build GAM Model Metrics
        _job.update(0, "Scoring training frame");
        scoreGenModelMetrics(model, train(), true); // score training dataset and generate model metrics
        if (valid() != null) {
          scoreGenModelMetrics(model, valid(), false); // score validation dataset and generate model metrics
        }
      } finally {
        try {
          final List<Key<Vec>> keep = new ArrayList<>();
          if (model != null) {
            if (_parms._keep_gam_cols) {
              keepFrameKeys(keep, newTFrame._key);
            } else {
              DKV.remove(newTFrame._key);
            }
          }
          if (dinfo != null)
            dinfo.remove();

          if (newValidFrame != null && validKeys != null) {
            keepFrameKeys(keep, newValidFrame._key);  // save valid frame keys for scoring later
            validKeys.addIfAbsent(newValidFrame._key);   // save valid frame keys from folds to remove later
            model._validKeys = validKeys;  // move valid keys here to model._validKeys to be removed later
          }
          Scope.exit(keep.toArray(new Key[keep.size()]));
        } finally {
          // Make sure Model is unlocked, as if an exception is thrown, the `ModelBuilder` expects the underlying model to be unlocked.
          model.update(_job);
          model.unlock(_job);
        }
      }
    }

    /**
     * This part will perform scoring and generate the model metrics for training data and validation data if 
     * provided by user.
     *      
     * @param model
     * @param scoreFrame
     * @param forTraining true for training dataset and false for validation dataset
     */
    private void scoreGenModelMetrics(GAMModel model, Frame scoreFrame, boolean forTraining) {
      Frame scoringTrain = new Frame(scoreFrame);
      model.adaptTestForTrain(scoringTrain, true, true);
      Frame scoredResult = model.score(scoringTrain);
      scoredResult.delete();
      ModelMetrics mtrain = ModelMetrics.getFromDKV(model, scoringTrain);
      if (mtrain!=null) {
        if (forTraining)
          model._output._training_metrics = mtrain;
        else 
          model._output._validation_metrics = mtrain;
        Log.info("GAM[dest="+dest()+"]"+mtrain.toString());
      } else {
        Log.info("Model metrics is empty!");
      }
    }

    GLMModel buildGLMModel(GAMParameters parms, Frame trainData, Frame validFrame) {
      GLMParameters glmParam = GamUtils.copyGAMParams2GLMParams(parms, trainData, validFrame);  // copy parameter from GAM to GLM
      if (_cv_lambda != null) { // use best alpha and lambda values from cross-validation to build GLM main model 
        glmParam._lambda = _cv_lambda;
        glmParam._alpha = _cv_alpha;
        glmParam._lambda_search = false;
      }
      int numGamCols = _parms._gam_columns.length;
      for (int find = 0; find < numGamCols; find++) {
        if ((_parms._scale != null) && (_parms._scale[find] != 1.0))
          _penalty_mat_center[find] = ArrayUtils.mult(_penalty_mat_center[find], _parms._scale[find]);
      }
      glmParam._glmType = gam;
      return new GLM(glmParam, _penalty_mat_center,  _gamColNamesCenter).trainModel().get();
    }

    void fillOutGAMModel(GLMModel glm, GAMModel model) {
      model._gamColNamesNoCentering = _gamColNames;  // copy over gam column names
      model._gamColNames = _gamColNamesCenter;
      model._output._gamColNames = _gamColNamesCenter;
      model._output._zTranspose = _zTranspose;
      model._gamFrameKeysCenter = _gamFrameKeysCenter;
      model._nclass = _nclass;
      model._output._binvD = _binvD;
      model._output._knots = _knots;
      model._output._numKnots = _numKnots;
      // extract and store best_alpha/lambda/devianceTrain/devianceValid from best submodel of GLM model
      model._output._best_alpha = glm._output.getSubmodel(glm._output._selected_submodel_idx).alpha_value;
      model._output._best_lambda = glm._output.getSubmodel(glm._output._selected_submodel_idx).lambda_value;
      model._output._devianceTrain = glm._output.getSubmodel(glm._output._selected_submodel_idx).devianceTrain;
      model._output._devianceValid = glm._output.getSubmodel(glm._output._selected_submodel_idx).devianceValid;
      model._gamColMeans = flat(_gamColMeans);
      if (_parms._lambda == null) // copy over lambdas used
        _parms._lambda = glm._parms._lambda.clone();
      if (_parms._keep_gam_cols)
        model._output._gam_transformed_center_key = model._output._gamTransformedTrainCenter.toString();
      if (_parms._savePenaltyMat) {
        model._output._penaltyMatrices_center = _penalty_mat_center;
        model._output._penaltyMatrices = _penalty_mat;
      }
      copyGLMCoeffs(glm, model, _parms, nclasses());  // copy over coefficient names and generate coefficients as beta = z*GLM_beta
      copyGLMtoGAMModel(model, glm, _parms, valid());  // copy over fields from glm model to gam model
    }
  }
}
