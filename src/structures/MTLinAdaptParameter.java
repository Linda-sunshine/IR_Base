package structures;

public class MTLinAdaptParameter {
	/***** Data sources.*****/
	public String m_data = "Amazon";// The default data set.
	public double m_adaptRatio = 0.5; // The ratio of data for training.
	
	/***** Parameters for regularizations.*****/
	public double m_eta1 = 1;
	public double m_eta2 = 0.5;
	public double m_eta3 = 0.1;
	public double m_eta4 = 0.3;
	
	/***** Selected Models for training(MTLinAdapt+Baseline models).*****/
	public String m_model = "mtlinbatch"; // Which model to use.
	public int m_fvGroup = 800; // The feature groups used for individual users.
	public int m_fvGroupSup = 5000; // The feature groups used for super user.
	
//	public int m_size = 400; // The size of users we want to use.
//	public int m_userSet = 1; // The set of users we want to use.
//	public int m_ttlSizeSet = 24; // The total number of sizes.
//	public int m_ttlUserSetNo = 10; // The total number of user sets.
	
	public MTLinAdaptParameter(String argv[])
	{
		int i;		
		// parse options
		for(i=0;i<argv.length;i++) {
			if(argv[i].charAt(0) != '-') 
				break;
			else if(++i>=argv.length)
				System.exit(1);
			else if (argv[i-1].equals("-data"))
				m_data = argv[i];
			else if( argv[i-1].equals("-adaptRatio"))
				m_adaptRatio = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-eta1"))
				m_eta1 = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-eta2"))
				m_eta2 = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-eta3"))
				m_eta3 = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-eta4"))
				m_eta4 = Double.valueOf(argv[i]);
//			else if (argv[i-1].equals("-size"))
//				m_size = Integer.valueOf(argv[i]);
//			else if (argv[i-1].equals("-set"))
//				m_userSet = Integer.valueOf(argv[i]);
//			else if (argv[i-1].equals("-userSetNo"))
//				m_ttlUserSetNo = Integer.valueOf(argv[i]);
//			else if (argv[i-1].equals("-sizeSetNo"))
//				m_ttlSizeSet = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-model"))
				m_model = argv[i];			
			else if (argv[i-1].equals("-fvgroup"))
				m_fvGroup = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-fvgroupsup"))
				m_fvGroupSup = Integer.valueOf(argv[i]);
			else
				exit_with_help();
		}
	}
	
	private void exit_with_help()
	{
		System.out.print(
		 "Usage: java execution [options] training_folder\n"
		+"--------------------------------------------------------------------------------\n"
		+"Parameters:\n"
		+"-data: specific the dataset used for training(default Amazon)\noption: Amazon, Yelp\n"
		+"-adaptRatio: the ratio of data for training: batch-0.5; online-1(default 0.5, online must be 1)\noption: (0, 1] for batch, 1 for online.\n"
		+"-eta1: coefficient for the scaling in each user's regularization(default 1)\n"
		+"-eta2: coefficient for the shifting in each user's regularization(default 0.5)\n"
		+"-eta3: coefficient for the scaling in super user's regularization(default 0.1)\n"
		+"-eta4: coefficient for the shifting in super user's regularization(default 0.3)\noption for eta1-eta4: (0, 1].\n"
		+"-model: specific training model,\noption: Base-base, GlobalSVM-gsvm, IndividualSVM-indsvm, RegLR-reglr, LinAdapt-linadapt, MultiTaskSVM-mtsvm, MTLinAdapt_Batch-mtlinbatch, MTLinAdapt_Online-mtlinonline\n"
		+"-fvgroup: feature groups for individual users(default 800),\noption: 400, 800, 1600, 5000\n"
		+"-fvgroupsup: feature groups for super user(default 5000),\noption: 400, 800, 1600, 5000\n"
		+"--------------------------------------------------------------------------------\n"
		);
		System.exit(1);
	}
}
