package structures;

public class MTLinAdaptParameter {
	public String m_data = "Amazon";
	public double m_eta1 = 1;
	public double m_eta2 = 0.5;
	public double m_lambda1 = 0.1;
	public double m_lambda2 = 0.3;
	public int m_size = 400; // The size of users we want to use.
	public int m_userSet = 1; // The set of users we want to use.
	public int m_ttlSizeSet = 24; // The total number of sizes.
	public int m_ttlUserSetNo = 10; // The total number of user sets.
	public String m_model = "mtlinadapt"; // Which model to use.
	
	public MTLinAdaptParameter(String argv[])
	{
		int i;		
		// parse options
		for(i=0;i<argv.length;i++) {
			if(argv[i].charAt(0) != '-') 
				break;
			else if(++i>=argv.length)
				System.exit(1);
			else if (argv[i-1].equals("-data")){
				System.out.println("data input.");
				m_data = argv[i];
			}
			else if (argv[i-1].equals("-eta1"))
				m_eta1 = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-eta2"))
				m_eta2 = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-lambda1"))
				m_lambda1 = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-lambda2"))
				m_lambda2 = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-size"))
				m_size = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-set"))
				m_userSet = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-userSetNo"))
				m_ttlUserSetNo = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-sizeSetNo"))
				m_ttlSizeSet = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-model"))
				m_model = argv[i];			
			else
				System.exit(1);
		}
	}
}
