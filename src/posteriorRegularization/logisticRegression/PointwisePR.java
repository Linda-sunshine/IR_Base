package posteriorRegularization.logisticRegression;


public class PointwisePR extends PosteriorConstraints {

	//pointwise constraint following Mustafizur's design; however, the semantics of this constraint is not well-defined or very weird
	public PointwisePR(double p[], int true_label, int label_size) { 
		super(p, label_size);
		
		parameters = new double[]{1.0};//start from a legal point
		gradient = new double[]{0.0};
		
		CONT_SIZE = 1;// pointwise constraint size
		
		initiate_constraint_feature(true_label);
	}

	@Override
	protected void initiate_constraint_feature(int label) {
		m_phi_Z_x = new double[C][CONT_SIZE];
		for(int i=0; i<C; i++)
			m_phi_Z_x[i][0] = -(4-Math.abs(i-label)); // distance to the true label
		
		m_b = new double[CONT_SIZE];
		m_q = new double[C];
	}
	
	@Override
	public String toString() {
		return "PointwisePR4LR";
	}
}
