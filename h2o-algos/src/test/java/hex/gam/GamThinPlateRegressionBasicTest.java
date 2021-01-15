package hex.gam;

import org.junit.Test;
import org.junit.runner.RunWith;
import water.DKV;
import water.Scope;
import water.TestUtil;
import water.fvec.Frame;
import water.runner.CloudSize;
import water.runner.H2ORunner;
import water.util.ArrayUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static hex.gam.GamSplines.ThinPlateRegressionUtils.*;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static water.util.ArrayUtils.sum;

@RunWith(H2ORunner.class)
@CloudSize(1)
public class GamThinPlateRegressionBasicTest extends TestUtil {
  @Test
  public void testKnotsDefault() {
    Scope.enter();
    try {
      Frame train = Scope.track(parse_test_file("smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv"));
      String[] ignoredCols = new String[]{"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"};
      String[][] gamCols = new String[][]{{"C6"},{"C7", "C8"}, {"C9"}};
      GAMModel.GAMParameters params = new GAMModel.GAMParameters();
      params._response_column = "C11";
      params._ignored_columns = ignoredCols;
      params._num_knots = new int[]{0,10,0};
      params._gam_columns = gamCols;
      params._train = train._key;
      GAMModel gam = new GAM(params).trainModel().get();
    } finally {
      Scope.exit();
    }
  }
  @Test
  public void testKnotsGenerationFromFrame() {
    Scope.enter();
    try {
      Frame knotsFrame1 = generate_real_only(1, 5, 0);
      final double[][] knots = new double[][]{{-1.9990569949269443}, {-0.9814307533427584}, {0.025991586992542004},
              {1.0077098743127828}, {1.999422899675758}};
      new ArrayUtils.CopyArrayToFrame(0,0,knots.length, knots).doAll(knotsFrame1);
      DKV.put(knotsFrame1);
      Scope.track(knotsFrame1);
      
      Frame knotsFrame2 = generate_real_only(2, 6, 0);
      final double[][] knots2 = new double[][]{{-1.99, -19.8}, {-0.98, -0.97}, {0.03, 0.031}, {0.05, 0.06},
              {1.01, 1.00}, {1.99, 1.98}};
      new ArrayUtils.CopyArrayToFrame(0,1,knots2.length, knots2).doAll(knotsFrame2);
      DKV.put(knotsFrame2);
      Scope.track(knotsFrame2);

      Frame knotsFrame3 = generate_real_only(1, 7, 0);
      final double[][] knots3 = new double[][]{{-1.9990569949269443}, {-0.9814307533427584}, {0.025991586992542004},
              {0.03}, {0.06}, {1.0077098743127828}, {1.999422899675758}};
      new ArrayUtils.CopyArrayToFrame(0,0,knots3.length, knots3).doAll(knotsFrame3);
      DKV.put(knotsFrame3);
      Scope.track(knotsFrame3);

      Frame train = Scope.track(parse_test_file("smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv"));
      String[] ignoredCols = new String[]{"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"};
      String[][] gamCols = new String[][]{{"C6"},{"C7", "C8"}, {"C9"}};
      GAMModel.GAMParameters params = new GAMModel.GAMParameters();
      params._knot_ids = new String[]{knotsFrame1._key.toString(), knotsFrame2._key.toString(), knotsFrame3._key.toString()};
      params._bs = new int[]{0,1,1};
      params._response_column = "C11";
      params._ignored_columns = ignoredCols;
      params._gam_columns = gamCols;
      params._train = train._key;
      GAMModel gam = new GAM(params).trainModel().get();
      Scope.track_generic(gam);
    } finally {
      Scope.exit();
    }
  }
  // For a given d, calculate m, then calculate the polynomial basis degree for each predictor involves.
  // However, the 0th order is not included at this stage.
  @Test
  public void testFindPolybasis() {
    int[] d = new int[]{1, 3, 5, 8, 10};
    int[] ans = new int[]{2, 10, 56, 495, 3003};
    for (int index = 0; index < d.length; index++) {
      int m = calculatem(d[index]);
      int[] polyOrder = new int[m];
      for (int tempIndex = 0; tempIndex < m; tempIndex++)
        polyOrder[tempIndex] = tempIndex;
      List<Integer[]> polyBasis = findPolybasis(d[index], m);
      assertEquals(ans[index], polyBasis.size()); // check and make sure number of basis is correct.  Content checked in testFindPermManyD already
      assertCorrectAllPerms(polyBasis, polyOrder);
    }
  }
  
  // given one combination, test that all permutations are returned
  @Test
  public void testFindAllPolybasis() {
    List<Integer[]> listOfCombos = new ArrayList<>();
    listOfCombos.add(new Integer[]{0, 0, 0, 0, 1}); // should get 5 permutations
    listOfCombos.add(new Integer[]{1, 2, 0, 0, 0}); // should get 20 permutations
    List<Integer[]> allCombos = findAllPolybasis(listOfCombos); // should be of size 5+20+1 (from all zeroes)
    assertEquals(26, allCombos.size()); // check correct size
    assertCorrectAllPerms(allCombos, new int[]{0,1,3}); // check correct content
  }
  
  
  public static void assertCorrectAllPerms(List<Integer[]> allCombos, int[] correctVals) {
    for (Integer[] oneList : allCombos) {
      int sumVal = sum(Arrays.stream(oneList).mapToInt(Integer::intValue).toArray());
      boolean correctSum = false;
      for (int val : correctVals)
        correctSum = correctSum || (sumVal == val);
      assertTrue(correctSum);
    }
  }
  
  @Test
  public void testFindPermManyD() {
    int[] d = new int[]{1, 3 ,5, 8, 10};
    int[] correctComboNum = new int[]{1, 3, 6, 11, 18};
    for (int index = 0; index < d.length; index++) {
      testFindPerm(d, correctComboNum, index);
    }
  }
  
  public void testFindPerm(int[] d, int[] correctComboNum, int testIndex) {
    int m = calculatem(d[testIndex]); // highest order of polynomial basis is m-1
    int[] totDegrees = new int[m];
    int[] degreeCombos = new int[m-1];
    for (int index = 0; index < totDegrees.length; index++)
      totDegrees[index] = index;
    int count = 0;
    for (int index = m-1; index > 0; index--) {
      degreeCombos[count++] = index;
    }

    // check for combos for totDegree = 0, 1, ..., m-1
    int numCombo = 0;
    for (int degree : totDegrees) {
      ArrayList<int[]> allCombos = new ArrayList<>();
      findOnePerm(degree, degreeCombos, 0, allCombos, null);
      assertCorrectPerm(allCombos, degree, degreeCombos);
      numCombo += allCombos.size();
    }
    assertEquals(numCombo, correctComboNum[testIndex]); // number of combos are correct
  }
  
  public static void assertCorrectPerm(ArrayList<int[]> allCombos, int degree, int[] degreeCombos) {
    for (int index = 0; index < allCombos.size(); index++) {
      int[] oneCombo = allCombos.get(index);
      int sum = 0;
      for (int tmpIndex = 0; tmpIndex < degreeCombos.length; tmpIndex++) {
        sum += oneCombo[tmpIndex]*degreeCombos[tmpIndex];
      }
      assertEquals(degree, sum);
    }
  }
  
  @Test
  public void testCalculatem() {
    int[] d = new int[]{1, 2, 3, 4, 10};
    int[] ans = new int[]{2, 2, 3, 3, 6};  // calculated by using (floor(d+1)/2)+1 from R
    
    for (int index = 0; index < d.length; index++) {
      int m = calculatem(d[index]);
      assertEquals(m, ans[index]);
    }
  }

  @Test
  public void testCalculateM() {
    int[] d = new int[]{1, 2, 3, 4, 5};
    int[] m = new int[]{2, 2, 3, 3, 4};  // calculated by using (floor(d+1)/2)+1 from R

    for (int index = 0; index < d.length; index++) {
      int M = calculateM(d[index], m[index]);
      assertEquals(M, factorial(d[index]+m[index]-1)/(factorial(d[index])*factorial(m[index]-1)));
    }
  }
  
  public int factorial(int n) {
    int prod = 1; 
    for (int index = 1; index <= n; index++)
      prod *= index;
    return prod;
  }
}
