<?php
echo "<body style='background-color:light'>";
$r = shell_exec("python dataset.py");
echo "$r\n";

?>