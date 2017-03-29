package structures;

public class DPParameter {
	
	public String m_prefix = "/if15/lg5bt/DataSigir"; //"./data/CoLinAdapt"
	public String m_data = "Amazon";
	public String m_model = "mtclinhdp";
	public int m_nuOfIterations = 30;
	public int m_M = 6;
	public int m_thinning = 5;
	public double m_sdA = 0.1;
	public double m_sdB = 0.1;
	
	// Concentration parameter
	public double m_alpha = 1;
	public double m_eta = 0.1;
	public double m_beta = 0.01;
	
	public double m_eta1 = 0.05;
	public double m_eta2 = 0.05;
	public double m_eta3 = 0.05;
	public double m_eta4 = 0.05;
	
	// MTCLRWithDP, MTCLRWithHDP
	public double m_q = 0.1; // global parameter.
	public double m_c = 1;// coefficient in front of language model weights.
	
	public int m_fv = 800;
	public int m_fvSup = 5000;
	
	// parameters for language models.7
	public String m_fs = "DF";
	public int m_lmTopK = 1000;
	public boolean m_post = false;
	
	public double m_adaptRatio = 0.5;
	// used in the sanity check of dp + x in testing.
	public int m_base = 30;
	public double m_th = 0.05;
	
	// used in the mixed model tuning.
	public int m_threshold = 15;
	
//	public int m_userSet = 10; // The set of users we want to use.
//	public int m_ttlSizeSet = 24; // The total number of sizes.
//	public int m_ttlUserSetNo = 10; // The total number of user sets.
	
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
			else if (argv[i-1].equals("-eta"))
				m_eta = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-beta"))
				m_beta = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-model"))
				m_model = argv[i];
			
			else if (argv[i-1].equals("-fv"))
				m_fv = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-fvSup"))
				m_fvSup= Integer.valueOf(argv[i]);
			
			else if (argv[i-1].equals("-M"))
				m_M = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-q"))
				m_q = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-lmc"))
				m_c = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-data"))
				m_data = argv[i];
			else if (argv[i-1].equals("-fs"))
				m_fs = argv[i];
			else if(argv[i-1].equals("-lmtopk"))
				m_lmTopK = Integer.parseInt(argv[i]);
			else if(argv[i-1].equals("-post"))
				m_post = Boolean.parseBoolean(argv[i]);
			else if(argv[i-1].equals("-base"))
				m_base = Integer.parseInt(argv[i]);
			else if(argv[i-1].equals("-th"))
				m_th = Double.parseDouble(argv[i]);
			else if(argv[i-1].equals("-adaptRatio"))
				m_adaptRatio = Double.parseDouble(argv[i]);
			else if(argv[i-1].equals("-threshold"))
				m_threshold = Integer.parseInt(argv[i]);
			else if(argv[i-1].equals("-prefix"))
				m_prefix = argv[i];
			else if(argv[i-1].equals("-thin"))
				m_thinning = Integer.valueOf(argv[i]);
			else
				exit_with_help();
		}
	}
	
	private void exit_with_help()
	{
		System.out.print(
				 "Usage: java execution [options] training_folder\n"
				+"options:\n"
				+"-fpath cv_file : list of controlled vocabular to be used in feature generation (default null)\n"
				+"-fstat fv_file : statistics of the features collected from the corpus (default null)\n"
				+"-vf vct_file : vector representation file (default null)\n"
				+"-dbf debug_file : classifier's debug output file (default null)\n"
				+"-fs type : feature selection method (default CHI)\n"
				+"	DF -- Document frequency\n"
				+"	CHI -- Chi-Square test statistics\n"
				+"	IG -- Informatoin gain\n"
				+"	MI -- Mutual information\n"
				+"-sp float : ignore the bottom proportion of ranked features (default 0.4)\n"
				+"-ep float : ignore the top proportion of ranked features (default 0.999)\n"
				+"-df int : ignore the features occurring less than c documents (default 10) \n"
				+"-cs int : total number of classes (has to be manually specified!)\n"
				+"-ngram int : n-gram for feature generation (default 2)\n"
				+"-lcut int : ignore the documents with length less than c (default 5)\n"
				+"-window int : window size in time series based sentiment analysis (default 0)\n"
				+"-cv int : cross validation fold (default 10)\n"
				+"-fv type : feature value generation method (default TFIDF)\n"
				+"	TF -- Term frequency\n"
				+"	TFIDF -- Term frequency times inverse document frequence\n"
				+"	BM25 -- Term frequency times BM25 IDF with document length normalization\n"
				+"	PLN -- Pivoted length normalization\n"
				+"-norm int : feature value normalization method (default L2)\n"
				+"	1 -- L1 normalization\n"
				+"	2 -- L2 normalization\n"
				+"	0 -- No normalization\n"
				+"-c type : classification method (default SVM)\n"
				+"	NB -- Naive Bayes\n"
				+"	LR -- Logistic Regression\n"
				+"	PR-LR -- Posterior Regularized Logistic Regression\n"
				+"	SVM -- Support Vector Machine (liblinear)\n"
				+"	GF -- Gaussian Fields by matrix inversion\n"
				+"	GF-RW -- Gaussian Fields by random walk\n"
				+"	GF-RW-ML -- Gaussian Fields by random walk with distance metric learning (by libliner)\n"
				+"	2topic -- Two-Topic Topic Model\n"
				+"	pLSA -- Probabilistic Latent Semantic Analysis\n"
				+"	gLDA -- Latent Dirichlet Allocation with Gibbs sampling\n"
				+"	vLDA -- Latent Dirichlet Allocation with variational inference\n"
				+"	HTMM -- Hidden Topic Markov Model\n"
				+"	LRHTMM -- MaxEnt Hidden Topic Markov Model\n"
				+"-cf type : multiple learning in Gaussian Fields (default SVM)\n"
				+"	NB -- Naive Bayes\n"
				+"	LR -- Logistic Regression\n"
				+"	PR-LR -- Posterior Regularized Logistic Regression\n"
				+"	SVM -- Support Vector Machine (liblinear)\n"
				+"-w type : instance weighting scheme (default None)\n"
				+"	PR -- Content similarity based PageRank\n"
				+"-s type : learning paradigm (default SUP)\n"
				+"	SUP -- Supervised learning\n"
				+"	SEMI -- Semi-supervised learning\n"
				+"	TM -- Topic Models\n"
				+"-C float -- trade-off parameter in LR and SVM (default 0.1)\n"
				+"-sr float : Sample rate for transductive learning (default 0.25)\n"
				+"-kUL int : k nearest labeled neighbors (default 100)\n"
				+"-kUU int : kP nearest unlabeled neighbors (default 50)\n"
				+"-wm 0/1 : weighted sum or majority vote in random walk\n"
				+"-sf 0/1 : use similiarity as weight in majority vote\n"
				+"-bd int : rating difference bound in generating pairwise constraint (default 3)\n"
				+"-csr double : constrain sampling rate for metric learning (default 1e-3)\n"
				+"-k int : number of topics (default 50)\n"
				+"-fprior prior_file : prior seed word list (default null)\n"
				+"-alpha float : dirichlet prior for p(z|d) (default 1.05)\n"
				+"-beta float : dirichlet prior for p(w|z) (default 1.01)\n"
				+"-lambda float : manual background proportion setting p(B) (default 0.8)\n"
				+"-eta float : random restart probability eta in randowm walk (default 0.1)\n"
				+"-gamma float : strength of prior (default 5.0)\n"
				+"-iter int : maximum number of EM iteration (default 100)\n"
				+"-con float : convergency limit for EM iterations (default 1e-5)\n"
				+"-viter int : maximum number of variational inference iteration (default 10)\n"
				+"-vcon float : convergency limit for variational inference iterations (default 1e-7)\n"
				+"-burn float : burn in period of sampling method (default 0.4)\n"
				+"-lag int : sampling lag when accumulating samples (default 10)\n"
				+"-mt 0/1 : using multi-thread for topic models (default 0)\n"	
				);
				System.exit(1);
			}
		System.exit(1);
	}
	
}
