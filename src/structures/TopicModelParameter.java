package structures;


public class TopicModelParameter {
	
	public String m_prefix = "/zf18/ll5fy/lab/dataset";//"./data/CoLinAdapt"
	public String m_source = "yelp"; // "amazon_movie"
	public String m_topicmodel = "ETBIR";

	public double m_beta = 1.0 + 1e-3;
	public double m_alpha = 1e-2;
	public double m_lambda = 1e-3;
	
	// model parameters for ETBIR
	public double m_sigma = 1.0 + 1e-2;
	public double m_rho = 1.0 + 1e-2;
	
	public int m_topk = 30;
	public int m_emIter = 100;
	public int m_number_of_topics = 30;
	public int m_varMaxIter = 20; // variational inference max iter number
	
	public double m_varConverge = 1e-5;
	public double m_emConverge = 1e-9;
	
	public String m_output = String.format("%s/%s/byUser_70k_review/output", m_prefix, m_source);// output directory
		
	public TopicModelParameter(String argv[]){
		
		int i;
		
		//parse options
		for(i=0;i<argv.length;i++) {
			if(argv[i].charAt(0) != '-') 
				break;
			else if(++i>=argv.length)
				System.exit(1);
			else if (argv[i-1].equals("-prefix"))
				m_prefix = argv[i];
			else if (argv[i-1].equals("-source"))
				m_source = argv[i];
			else if(argv[i-1].equals("-topicmodel"))
				m_topicmodel = argv[i];
			
			else if(argv[i-1].equals("-alpha"))
				m_alpha = Double.valueOf(argv[i]);
			else if(argv[i-1].equals("-beta"))
				m_beta = Double.valueOf(argv[i]);
			else if(argv[i-1].equals("-lambda"))
				m_lambda = Double.valueOf(argv[i]);
			else if(argv[i-1].equals("-sigma"))
				m_sigma = Double.valueOf(argv[i]);
			else if(argv[i-1].equals("-rho"))
				m_rho = Double.valueOf(argv[i]);
			
			else if(argv[i-1].equals("-topk"))
				m_topk = Integer.valueOf(argv[i]);
			else if(argv[i-1].equals("-emIter"))
				m_emIter = Integer.valueOf(argv[i]);
			else if(argv[i-1].equals("-nuOfTopics"))
				m_number_of_topics = Integer.valueOf(argv[i]);
			else if(argv[i-1].equals("-varMaxIter"))
				m_varMaxIter = Integer.valueOf(argv[i]);
			
			else if(argv[i-1].equals("-varConverge"))
				m_varConverge = Double.valueOf(argv[i]);
			else if(argv[i-1].equals("-emConverge"))
				m_emConverge = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-output"))
				m_output = argv[i];
			else
				System.exit(1);
		}
	}
}