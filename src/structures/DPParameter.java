package structures;

public class DPParameter {
	public String m_data = "Amazon";
	public String m_model = "mtclindp";
	public int m_nuOfIterations = 15;
	public int m_M = 6;
	public double m_sdA = 0.2;
	public double m_sdB = 0.2;
	public double m_alpha = 1;
	public double m_eta1 = 0.05;
	public double m_eta2 = 0.05;
	public double m_eta3 = 0.05;
	public double m_eta4 = 0.05;
	
	public int m_userSet = 10; // The set of users we want to use.
	public int m_ttlSizeSet = 24; // The total number of sizes.
	public int m_ttlUserSetNo = 10; // The total number of user sets.
	
	public DPParameter(String argv[]){
		
		int i;
		
		//parse options
		for(i=0;i<argv.length;i++) {
			if(argv[i].charAt(0) != '-') 
				break;
			else if(++i>=argv.length)
				exit_with_help();
			else if (argv[i-1].equals("-nuI"))
				m_nuOfIterations = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-sdA"))
				m_sdA = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-sdB"))
				m_sdB = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-eta1"))
				m_eta1 = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-eta2"))
				m_eta2 = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-eta3"))
				m_eta3 = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-eta4"))
				m_eta4 = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-alpha"))
				m_alpha = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-model"))
				m_model = argv[i];
			else if (argv[i-1].equals("-M"))
				m_M = Integer.valueOf(argv[i]);
			else
				exit_with_help();
		}
	}
	
	private void exit_with_help()
	{
		System.exit(1);
	}
	
}
