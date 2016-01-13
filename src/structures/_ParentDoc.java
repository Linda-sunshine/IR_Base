package structures;

import java.util.ArrayList;

public class _ParentDoc extends _Doc {

	public ArrayList<_ChildDoc> m_childDocs;

	public _ParentDoc(int ID, String name, String title, String source, int ylabel) {
		super(ID, source, ylabel);

		m_childDocs = new ArrayList<_ChildDoc>();

		setName(name);
		setTitle(title);
	}
	
	public void addChildDoc(_ChildDoc cDoc){
		m_childDocs.add(cDoc);
	}
}
