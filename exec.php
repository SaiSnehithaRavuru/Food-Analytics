
<?php
    #echo "<body style='background-color:cyan'>";

	if(isset($_POST['submit']))
	{
		$date= $_POST['datepicker'];
		#echo $date;

		$arr = $date;
		$resStr = str_replace('/', '-', $arr);
		#print_r($resStr);
		$OriginalString = $resStr;
	
		$arr = explode("-",$OriginalString);

 		$res = ($arr[1])."-".($arr[0])."-".($arr[2]);
		print_r($res);

		$r = shell_exec("python consumptionstats.py $res");
		echo "$r\n";
	}
	

?>    
