package posteriorRegularization.logisticRegression;


public class PairwisePR extends PosteriorConstraints {

	public PairwisePR(double p[], int true_label, int label_size) {//HdT means the difference between head and tail 
		super(p, label_size);
		
		parameters = new double[]{1.0, 1.0, 1.0, 1.0};//start from a legal point
		gradient = new double[]{0.0, 0.0, 0.0, 0.0};
		CONT_SIZE = 4;// pairwise constraint size
		
		initiate_constraint_feature(true_label);
	}

	@Override
	protected void initiate_constraint_feature(int label) {
		m_phi_Z_x = new double[C][CONT_SIZE];
		m_b = new double[CONT_SIZE];
		m_q = new double[C];
		
		//Arrays.fill(m_b, 5e-2);
		
		if(label == 0) {
			m_phi_Z_x[0][0] = -1;
			m_phi_Z_x[1][0] = 1;
			
			m_phi_Z_x[1][1] = -1;
			m_phi_Z_x[2][1] = 1;
			
			m_phi_Z_x[2][2] = -1;
			m_phi_Z_x[3][2] = 1;
			
			m_phi_Z_x[3][3] = -1;
			m_phi_Z_x[4][3] = 1;
		} else if(label == 1) {
			m_phi_Z_x[0][0] = 1;
			m_phi_Z_x[1][0] = -1;
			
			m_phi_Z_x[1][1] = -1;
			m_phi_Z_x[2][1] = 1;
			
			m_phi_Z_x[2][2] = -1;
			m_phi_Z_x[3][2] = 1;
			
			m_phi_Z_x[3][3] = -1;
			m_phi_Z_x[4][3] = 1;
		} else if(label == 2) {
			m_phi_Z_x[0][0] = 1;
			m_phi_Z_x[1][0] = -1;
			
			m_phi_Z_x[1][1] = 1;
			m_phi_Z_x[2][1] = -1;
			
			m_phi_Z_x[2][2] = -1;
			m_phi_Z_x[3][2] = 1;
			
			m_phi_Z_x[3][3] = -1;
			m_phi_Z_x[4][3] = 1;
		} else if(label == 3) {
			m_phi_Z_x[0][0] = 1;
			m_phi_Z_x[1][0] = -1;
			
			m_phi_Z_x[1][1] = 1;
			m_phi_Z_x[2][1] = -1;
			
			m_phi_Z_x[2][2] = 1;
			m_phi_Z_x[3][2] = -1;
			
			m_phi_Z_x[3][3] = -1;
			m_phi_Z_x[4][3] = 1;
		} else if(label == 4) {
			m_phi_Z_x[0][0] = 1;
			m_phi_Z_x[1][0] = -1;
			
			m_phi_Z_x[1][1] = 1;
			m_phi_Z_x[2][1] = -1;
			
			m_phi_Z_x[2][2] = 1;
			m_phi_Z_x[3][2] = -1;
			
			m_phi_Z_x[3][3] = 1;
			m_phi_Z_x[4][3] = -1;
		}
	}
	
	@Override
	public String toString() {
		return "PairwisePR4LR";
	}
}
