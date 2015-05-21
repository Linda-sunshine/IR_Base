package posteriorRegularization.logisticRegression;

import optimization.gradientBasedMethods.ProjectedObjective;
import optimization.projections.BoundsProjection;
import optimization.projections.Projection;
import utils.Utils;

abstract public class PosteriorConstraints extends ProjectedObjective {

	protected double[] m_p, m_q, m_b;	
	protected double[][] m_phi_Z_x;
	
	double m_epsilon = 0.1; // slack variable
	protected Projection m_projection;
	
	protected int C = 5; // class size, this should be treated as an input!!!
	protected int CONT_SIZE;
	
	public PosteriorConstraints(double p[], int label_size) {
		m_p = p;
		C = label_size;		
		m_projection = new BoundsProjection(0.0, Double.MAX_VALUE);
	}
	
	abstract protected void initiate_constraint_feature(int label);
	
	// this is normalized posterior
	public double[] getPosterior() {
		calcPosterior();
		Utils.scaleArray(m_q, 1.0/Utils.sumOfArray(m_q));
		return m_q;
	}
	
	// this is unnormalized posterior
	void calcPosterior() {
		getPosteriorScaler(m_q);
		for(int i=0; i<C; i++)
			m_q[i] *= m_p[i];
	}
	
	public void getPosteriorScaler(double[] scaler) {
		for(int i=0; i<C; i++) {
			double sum = 0.0;
			for(int j=0; j<CONT_SIZE;j++)
				sum -= parameters[j] * m_phi_Z_x[i][j];
			scaler[i] = Math.exp(sum);
		}
	}
	
	@Override
	public double[] getGradient() {
		gradientCalls ++;
		calcPosterior();
		
		double z_lambda = Utils.sumOfArray(m_q);
		for(int i=0; i<CONT_SIZE; i++) {
			gradient[i] = 2*m_epsilon*parameters[i] + m_b[i];
			for(int k=0; k<C; k++)
				gradient[i] -= m_phi_Z_x[k][i]*m_q[k]/z_lambda;
		}
		return gradient;
	}

	@Override
	public double getValue() {
		functionCalls++;
		calcPosterior();
		
		return Math.log(Utils.sumOfArray(m_q)) + m_epsilon*Utils.dotProduct(parameters, parameters) + Utils.dotProduct(m_b, parameters);
	}
	
	@Override
	public double[] projectPoint(double[] point) {
		double[] newPoint = point.clone();
		m_projection.project(newPoint);
		return newPoint;
	}
}
