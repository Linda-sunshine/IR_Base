package structures;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 */
public class EmbeddingParameter {

    public String m_prefix = "/zf8/lg5bt/DataSigir";//"./data/CoLinAdapt"
    public String m_data = "YelpNew";
    public String m_savePrefix = "/zf8/lg5bt/embeddingExp/eub/";

    public int m_emIter = 50;
    public int m_number_of_topics = 10;
    public int m_varIter = 10;
    public int m_embeddingDim = 10;
    public int m_kFold = 0;

    public boolean m_multiFlag = true;
    public double m_stepSize = 1e-3;

    public boolean m_alphaFlag = false;
    public boolean m_gammaFlag = true;
    public boolean m_betaFlag = true;
    public boolean m_tauFlag = false;
    public boolean m_xiFlag = false;
    public boolean m_ada = false;

    public EmbeddingParameter(String argv[]) {

        int i;

        //parse options
        for (i = 0; i < argv.length; i++) {
            if (argv[i].charAt(0) != '-')
                break;
            else if (++i >= argv.length)
                exit_with_help();
            else if (argv[i - 1].equals("-prefix"))
                m_prefix = argv[i];
            else if (argv[i - 1].equals("-data"))
                m_data = argv[i];
            else if (argv[i - 1].equals("-emIter"))
                m_emIter = Integer.valueOf(argv[i]);
            else if (argv[i - 1].equals("-nuTopics"))
                m_number_of_topics = Integer.valueOf(argv[i]);
            else if (argv[i - 1].equals("-varIter"))
                m_varIter = Integer.valueOf(argv[i]);
            else if (argv[i - 1].equals("-dim"))
                m_embeddingDim = Integer.valueOf(argv[i]);
            else if (argv[i - 1].equals("-multi"))
                m_multiFlag = Boolean.valueOf(argv[i]);
            else if (argv[i - 1].equals("-stepSize"))
                m_stepSize = Double.valueOf(argv[i]);
            else if (argv[i - 1].equals("-savePrefix"))
                m_savePrefix = argv[i];
            else if (argv[i - 1].equals("-alphaFlag"))
                m_alphaFlag = Boolean.valueOf(argv[i]);
            else if (argv[i - 1].equals("-gammaFlag"))
                m_gammaFlag = Boolean.valueOf(argv[i]);
            else if (argv[i - 1].equals("-betaFlag"))
                m_betaFlag = Boolean.valueOf(argv[i]);
            else if (argv[i - 1].equals("-tauFlag"))
                m_tauFlag = Boolean.valueOf(argv[i]);
            else if (argv[i - 1].equals("-xiFlag"))
                m_xiFlag = Boolean.valueOf(argv[i]);
            else if (argv[i - 1].equals("-kFold"))
                m_kFold = Integer.valueOf(argv[i]);
            else if (argv[i - 1].equals("-ada"))
                m_ada = Boolean.valueOf(argv[i]);

        }
    }

    private void exit_with_help() {
        System.exit(1);
    }
}
