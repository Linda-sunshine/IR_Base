package structures;

public class CFParameter {
	
	public String m_data = "Amazon";
	public int m_t = 2;
	public int m_k = 6;
	
	public CFParameter(String argv[]){
		
		int i;
		
		//parse options
		for(i=0;i<argv.length;i++) {
			if(argv[i].charAt(0) != '-') 
				break;
			else if(++i>=argv.length)
				exit_with_help();
			else if (argv[i-1].equals("-t"))
				m_t = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-k"))
				m_k = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-data"))
				m_data = argv[i];
			else
				exit_with_help();
		}
	}
	
	private void exit_with_help()
	{
		System.exit(1);
	}
	
}
