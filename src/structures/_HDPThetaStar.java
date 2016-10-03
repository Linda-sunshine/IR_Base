package structures;
/**
 * The structure wraps both \phi(_thetaStar) and \psi.
 * @author lin
 *
 */
public class _HDPThetaStar extends _thetaStar {
	
	// beta in _thetaStar is \phi used in HDP.
	double[] m_psi;// psi used in multinomal distribution.
	public int m_hSize; //total number of local groups in the component.
	
	public _HDPThetaStar(int dim) {
		super(dim);
	}

	public double[] getPsiModel(){
		return m_psi;
	}
}
