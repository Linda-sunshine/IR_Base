package CoLinAdapt;

import java.util.ArrayList;
import java.util.TreeMap;

import structures._Review;
import structures._User;

public class OneCoLinAdapt extends OneLinAdapt{

	public OneCoLinAdapt(_User u, int fg, int fn, TreeMap<Integer, ArrayList<Integer>> featureGroupIndex) {
		super(u, fg, fn);
	}
	
	// Calculate the new function value.
	public double calculateFunctionValue(ArrayList<_Review> trainSet){
		double fValue = 0;
		
		return fValue;
	}
	
	//Calculate the gradients for the use in LBFGS.
	public void calculateGradients(ArrayList<_Review> trainSet){
		
	}
	
}
