package structures;

public class CoLinAdaptParameter {
	public double m_eta1 = 0.5; // Coefficient for shift.
	public double m_eta2 = 0.5; // Coefficient for scale.
	public double m_eta3 = 0.5; // Coefficient for R2.
	public int m_topK = 20; // Select topK users as neighbors.
	
	public CoLinAdaptParameter(String argv[]){
		
		int i;
		
		//parse options
		for(i=0;i<argv.length;i++) {
			if(argv[i].charAt(0) != '-') 
				break;
			else if(++i>=argv.length)
				exit_with_help();
			else if (argv[i-1].equals("-eta1"))
				m_eta1 = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-eta2"))
				m_eta2 = Double.valueOf(argv[i]);
			else if(argv[i-1].equals("-eta3"))
				m_eta3 = Double.valueOf(argv[i]);
			else if(argv[i-1].equals("-topK"))
				m_topK = Integer.valueOf(argv[i]);
			else
				exit_with_help();
		}
	}
		
	private void exit_with_help()
	{
		System.exit(1);
	}
	
	public void printParameters(){
		System.out.print("----------------------Parameters for CoLinAdapt------------------------\n");
		System.out.format("shift: %.4f\tScale: %.4f\tCoefficient for R2: %.4f\tTopK: %d\n", m_eta1, m_eta2, m_eta3, m_topK);
		System.out.print("-----------------------------------------------------------------------\n");
	}
}
