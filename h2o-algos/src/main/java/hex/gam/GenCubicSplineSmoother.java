package hex.gam;

import hex.gam.MatrixFrameUtils.GamUtils;
import hex.gam.MatrixFrameUtils.GenerateGamMatrixOneColumn;
import jsr166y.RecursiveAction;
import water.DKV;
import water.Key;
import water.fvec.Frame;
import water.fvec.Vec;
import water.util.ArrayUtils;

public class GenCubicSplineSmoother extends RecursiveAction {
  
  
  @Override
  protected void compute() {
      GenerateGamMatrixOneColumn genOneGamCol = new GenerateGamMatrixOneColumn(splineType, numKnots,
              _knots[frameIndex][0], predictVec).doAll(numKnots, Vec.T_NUM,predictVec);
      if (_parms._savePenaltyMat)  // only save this for debugging
        GamUtils.copy2DArray(genOneGamCol._penaltyMat, _penalty_mat[frameIndex]); // copy penalty matrix
      // calculate z transpose
      Frame oneAugmentedColumnCenter = genOneGamCol.outputFrame(Key.make(), newColNames,
              null);
      for (int cind=0; cind < numKnots; cind++)
        _gamColMeans[frameIndex][cind] = oneAugmentedColumnCenter.vec(cind).mean();
      oneAugmentedColumnCenter = genOneGamCol.centralizeFrame(oneAugmentedColumnCenter,
              predictVec.name(0) + "_" + splineType + "_center_", _parms);
      GamUtils.copy2DArray(genOneGamCol._ZTransp, _zTranspose[frameIndex]); // copy transpose(Z)
      double[][] transformedPenalty = ArrayUtils.multArrArr(ArrayUtils.multArrArr(genOneGamCol._ZTransp,
              genOneGamCol._penaltyMat), ArrayUtils.transpose(genOneGamCol._ZTransp));  // transform penalty as zt*S*z
      GamUtils.copy2DArray(transformedPenalty, _penalty_mat_center[frameIndex]);
      _gamFrameKeysCenter[frameIndex] = oneAugmentedColumnCenter._key;
      DKV.put(oneAugmentedColumnCenter);
      System.arraycopy(oneAugmentedColumnCenter.names(), 0, _gamColNamesCenter[frameIndex], 0,
              numKnotsM1);
      GamUtils.copy2DArray(genOneGamCol._bInvD, _binvD[frameIndex]);
      _numKnots[frameIndex] = genOneGamCol._numKnots;
  }
}
