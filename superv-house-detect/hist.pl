#!/usr/bin/perl
use Scalar::Util qw(looks_like_number);
use POSIX;

$count=shift @ARGV;
while(<>) {
  chomp;
  push @data, $_ if looks_like_number($_);
}

$min = $data[0];
$max = $data[0];

foreach $_ (@data) {
  $min = $_ if $_ < $min;
  $max = $_ if $_ > $max;
}

#print $min." ".$max."\n";

foreach $_ (@data) {
  $bin = floor(($_ - $min)/($max - $min)*$count);
  $bin-- if ($bin == $count);
  $hist[$bin]++;
}

print "  \t".$min."\n";
for($i = 0; $i <= $#hist; $i++) {
  print $i."\t".($min + ($i + 1)*($max - $min)/$count)."\t".$hist[$i]."\n" if $hist[$i] > 0;
}
