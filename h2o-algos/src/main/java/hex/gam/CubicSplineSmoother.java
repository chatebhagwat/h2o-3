package hex.gam;

import hex.gam.MatrixFrameUtils.GamUtils;
import hex.gam.MatrixFrameUtils.GenerateGamMatrixOneColumn;
import jsr166y.RecursiveAction;
import water.DKV;
import water.Key;
import water.fvec.Frame;
import water.fvec.Vec;
import water.util.ArrayUtils;
import static hex.gam.GAMModel.GAMParameters;
import static hex.gam.MatrixFrameUtils.GamUtils.AllocateType;
import static hex.gam.MatrixFrameUtils.GamUtils.allocate2DArray;

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
  public String[] _gamColNamesCenter;
  public double[][] _zTranspose;
  public double[][] _penalty_mat;
  public double[][] _penalty_mat_center;
  public double[][] _binvD;
  public Key<Frame> _gamFrameKeysCenter;
  public double[] _gamColMeans;

  
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
    _penalty_mat = new double[_numKnots][_numKnots];
    _penalty_mat_center = new double[_numKnotsM1][_numKnotsM1];
    _fileMode = fileM;
    _binvD = allocate2DArray(parms, _fileMode, _numKnots);
    _gamColNamesCenter = new String[_numKnotsM1];
    _gamColMeans = new double[_numKnots];
  }
  
  @Override
  protected void compute() {
      GenerateGamMatrixOneColumn genOneGamCol = new GenerateGamMatrixOneColumn(_splineType, _numKnots,
              _knots, _predictVec).doAll(_numKnots, Vec.T_NUM, _predictVec);
      if (_savePenaltyMat)  // only save this for debugging
        GamUtils.copy2DArray(genOneGamCol._penaltyMat, _penalty_mat); // copy penalty matrix
      Frame oneAugmentedColumnCenter = genOneGamCol.outputFrame(Key.make(), _newColNames,
              null);
      oneAugmentedColumnCenter = genOneGamCol.centralizeFrame(oneAugmentedColumnCenter,
              _predictVec.name(0) + "_" + _splineType + "_center_", _parms);
      _zTranspose = genOneGamCol._ZTransp;
      double[][] transformedPenalty = ArrayUtils.multArrArr(ArrayUtils.multArrArr(genOneGamCol._ZTransp,
              genOneGamCol._penaltyMat), ArrayUtils.transpose(genOneGamCol._ZTransp));  // transform penalty as zt*S*z
      GamUtils.copy2DArray(transformedPenalty, _penalty_mat_center);
      _gamFrameKeysCenter = oneAugmentedColumnCenter._key;
      DKV.put(oneAugmentedColumnCenter);
    System.arraycopy(oneAugmentedColumnCenter.names(), 0, _gamColNamesCenter, 0,
            _numKnotsM1);
      GamUtils.copy2DArray(genOneGamCol._bInvD, _binvD);
  }
}
