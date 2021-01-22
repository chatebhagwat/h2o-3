package hex.gam.GamSplines;

import water.MRTask;
import water.MemoryManager;
import water.fvec.Chunk;
import water.fvec.NewChunk;

import java.util.List;

import static hex.gam.GamSplines.ThinPlatePolynomialBasisUtils.*;

public class ThinPlatePolynomialWithKnots extends MRTask<ThinPlatePolynomialWithKnots> {
  final int _weightID;
  final List<Integer[]> _polyBasisList;
  final int _M; // size of polynomial basis
  final int _d; // number of predictors used
  
  public ThinPlatePolynomialWithKnots(int weightID, List<Integer[]> polyBasis) {
    _weightID = weightID;
    _polyBasisList = polyBasis;
    _M = polyBasis.size();
    _d = weightID;
  }

  @Override
  public void map(Chunk[] chk, NewChunk[] newGamCols) {
    int numRow = chk[0].len();
    double[] onePolyRow = MemoryManager.malloc8d(_M);
    double[] oneDataRow = MemoryManager.malloc8d(_d);
    for (int rowIndex = 0; rowIndex < numRow; rowIndex++) {
      if (chk[_weightID].atd(rowIndex) != 0) {
        if (checkRowNA(chk, rowIndex)) {
          fillRowOneValue(newGamCols, _M, Double.NaN);
        } else {
          calculatePolynomialBasis(onePolyRow, oneDataRow, chk, rowIndex);
          fillRowArray(newGamCols, _M, onePolyRow);
        }
      } else {  // set the row to zero
        fillRowOneValue(newGamCols, _M, 0.0);
      }
    }
  }
  
  void calculatePolynomialBasis(double[] onePolyRow, double[] oneDataRow, Chunk[] chk, int rowIndex) {
    extractOneRowFromChunk(chk, rowIndex, oneDataRow);
    for (int colIndex = 0; colIndex < _M; colIndex++) {
      Integer[] oneBasis = _polyBasisList.get(colIndex);
      double val = 1.0;
      for (int predIndex = 0; predIndex < _d; predIndex++) {
        val *= Math.pow(oneDataRow[predIndex], oneBasis[predIndex]);
      }
      onePolyRow[colIndex] = val;
    }
  }
  
  void extractOneRowFromChunk(Chunk[] chk, int rowIndex, double[] oneRow) {
    for (int colInd = 0; colInd < _d; colInd++)
      oneRow[colInd] = chk[colInd].atd(rowIndex);
  }
}
