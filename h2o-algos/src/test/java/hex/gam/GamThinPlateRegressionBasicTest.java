package hex.gam;

import org.junit.Test;
import org.junit.runner.RunWith;
import water.TestUtil;
import water.runner.CloudSize;
import water.runner.H2ORunner;

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
