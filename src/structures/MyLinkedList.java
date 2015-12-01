package structures;

import java.util.LinkedList;

public class MyLinkedList<E extends Comparable<? super E>> extends LinkedList<E> {
//	protected E m_head;
	
	public MyLinkedList(){
		super();
	}
	
	//Add one element to the linked list ordered by the time stamps.
	public boolean add(E object){
		for(int i=0; i<super.size(); i++){
			if(object.compareTo(super.get(i))==-1){//object.val < p.val
				super.add(i, object);
				return true;
			}
		}
		super.addLast(object);
		return true;// If the inserted element is the largest, put it in the end.
	}

	public static void main(String[] args){
//		MyLinkedList<_Review> test = new MyLinkedList<_Review>();
//		_Review d0 = new _Review(0, "good", 0, "jack", (long) 100);
//		_Review d1 = new _Review(1, "bad", 1, "mary", (long) 50);
//		_Review d2 = new _Review(2, "great", 1, "jenny", (long) 200);
//		test.addFirst(d0);
//		test.add(d1);
//		test.add(d2);
	}	
}
