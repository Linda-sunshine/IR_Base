package CoLinAdapt;

import java.util.ArrayList;

import structures._Review;
import structures._User;

public class CoLinAdapt extends LinAdapt{
	double m_eta3; // Weight for R2.
	ArrayList<_User> m_neighbors;
	
	public CoLinAdapt(int fg, int fn, double[] globalWeights, int[] featureGroupIndexes) {
		super(fg, fn, globalWeights, featureGroupIndexes);
	}
	
	public void setNeighbors(ArrayList<_User> neighbors){
		m_neighbors = neighbors;
	}
	// Calculate the new function value.
	public double calculateFunctionValue(ArrayList<_Review> trainSet){
		double fValue = super.calculateFunctionValue(trainSet);
		// Add the R2 part to the function value.
		double R2 = 0, sim;
		double[] A;
		_User neighbor;
		for(int i=0; i<m_neighbors.size(); i++){
			neighbor = m_neighbors.get(i);
			A = neighbor.getCoLinAdapt().getA();
			sim = calculateSim(m_neighbors.get(i));
			for(int j=0; j<m_dim; j++){
				//(a_ki-a_kj)^2 + (b_ki-b_kj)^2
				R2 += (m_A[j] - A[j])*(m_A[j] - A[j]) + (m_A[j + m_dim] - A[j + m_dim])*(m_A[j + m_dim] - A[j + m_dim]) ;
			}
			fValue -= sim * m_eta3 * R2;
		}
		return fValue ;
	}
	
	// Calculate the similarity between two users.
	public double calculateSim(_User u){
		double sim = 0;
		/****
		 * Need to be filled.
		 */
		return sim;
	}
	//Calculate the gradients for the use in LBFGS.
	public void calculateGradients(ArrayList<_Review> trainSet){
		/****
		 * Need to be filled.
		 */
	}
	
}
