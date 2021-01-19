package hex.gam.GamSplines;

import water.MRTask;
import water.MemoryManager;
import water.fvec.Chunk;
import water.fvec.NewChunk;

import static hex.gam.GamSplines.ThinPlatePolynomialBasisUtils.calculatem;
import static org.apache.commons.math3.util.CombinatoricsUtils.factorial;

// Implementation details of this class can be found in GamThinPlateRegressionH2O.doc attached to this 
// JIRA: https://h2oai.atlassian.net/browse/PUBDEV-7860
public class ThinPlateDistanceWithKnots extends MRTask<ThinPlateDistanceWithKnots> {
  final double[][] _knots;  // store knot values for the spline class
  final int _knotNum; // number of knot values
  final int _d; // number of predictors for smoothers
  final int _m; // highest degree of polynomial basis +1
  final double _constantTerms;
  final int _weightID;
  final boolean _dEven;
  final int _numPred;
  final int _distancePower;
  
  public ThinPlateDistanceWithKnots(double[][] knots, int d) {
    _knots = knots;
    _knotNum = _knots.length;
    _d = d;
    _dEven = _d/2==0;
    _m = calculatem(_d);
    _weightID = _d; // weight column index
    _numPred = _knots[0].length;
    _distancePower = 2*_m-_d;
    if (d/2 == 0)
      _constantTerms = Math.pow(-1, _m+1+_d/2)/(Math.pow(2, _m-1)*Math.pow(Math.PI, _d/2)*factorial(_m-_d/2));
    else
      _constantTerms = Math.pow(-1, _m)*_m/(factorial(2*_m)*Math.pow(Math.PI, (_d-1)/2));
  }

  @Override
  public void map(Chunk[] chk, NewChunk[] newGamCols) {
    int nrows = chk[0].len();
    double[] rowValues = MemoryManager.malloc8d(_knotNum);
    for (int rowIndex = 0; rowIndex < nrows; rowIndex++) {
      if (chk[_weightID].atd(rowIndex) != 0) {
        if (chk[_weightID].hasNA()) {
          for (int colInd = 0; colInd < _knotNum; colInd++) // insert NaNs for rows containing NaN
            newGamCols[colInd].addNum(Double.NaN);
        } else {  // calculate distance measure as in 3.1
           calculateDistance(rowValues, chk[rowIndex]);
           for (int colInd = 0; colInd < _knotNum; colInd++)
             newGamCols[colInd].addNum(rowValues[colInd]);
        }
        
      } else {  // insert 0 to newChunk for weigth == 0
        for (int colInd = 0; colInd < _knotNum; colInd++)
          newGamCols[colInd].addNum(0.0);
      }
    }
  }
  
  void calculateDistance(double[] rowValues, Chunk oneRow) { // see 3.1
    for (int knotInd = 0; knotInd < _knotNum; knotInd++) {
      double sumSq = 0;
      for (int predInd = 0; predInd < _numPred; predInd++)
        sumSq += oneRow.atd(predInd)*oneRow.atd(predInd)-_knots[knotInd][predInd]*_knots[knotInd][predInd];

      double distance = Math.sqrt(sumSq);
      rowValues[knotInd] = _constantTerms*distance;
      if (_dEven)
        rowValues[knotInd] *= Math.log(distance);
    }
  }
}
