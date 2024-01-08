
public class Q10 {

	public static void main(String[] args) {
		String s = "123";
		combination("",s);
		}
	
	public static void combination(String up, String op) {
		if(op == "") {
			System.out.println(up);
			 return;
		}
		String c = op.charAt(0) + "";
		int s = up.length() ;
		for (int i =0 ; i<=s;i++) {
			String s1 = up.substring(0,i);
			String s2 = up.substring(i,s);
			combination(s1+c+s2  , op.substring(1));
		}
		
	}
	}
