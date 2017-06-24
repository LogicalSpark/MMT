# $Id$
use warnings;
use strict;

binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";

my $lowercase = 0; # by default, the casing of hyp and refs is kept as it is
my $tokenization = 1; # by default, hyp and refs are tokenized
my $stem = ();
my $HELP = 0;
my $order = 4;

while (@ARGV) {
        $_ = shift(@ARGV);
        /^[^-]/     && ($stem=$_, last);            # file of references
        /^-lc$/     && ($lowercase = 1, next);      # lowercasing hyp and refs
        /^-nt$/     && ($tokenization = 0, next);   # hyp and refs are not tokenized
        /^-order$/  && ($order = shift(@ARGV), next);       # order of the ngram for BELU (default=4)
        /^-h/       && ($HELP = 1, last);
}

if (!defined $stem || $HELP) {
  print STDERR "usage: mmt-bleu.pl [-lc] [-nt] [-order INT] reference < hypothesis\n";
  print STDERR "Reads the references from reference or reference0, reference1, ...\n";
  exit(1);
}

if ($order == 0){
  print STDERR "The order of the bleu score should be larger than 0\n";
  exit(1);
}

$stem .= ".ref" if !-e $stem && !-e $stem."0" && -e $stem.".ref0";

my @REF;
my $ref=0;
while(-e "$stem$ref") {
    &add_to_ref("$stem$ref",\@REF);
    $ref++;
}
&add_to_ref($stem,\@REF) if -e $stem;
die("ERROR: could not find reference file $stem") unless scalar @REF;

# add additional references explicitly specified on the command line
shift;
foreach my $stem (@ARGV) {
    &add_to_ref($stem,\@REF) if -e $stem;
}

sub tokenization_international {
  my ($norm_text) = @_;
  
  my $TAG_NAME_RX = '(\p{Alpha}|_|:)(\p{Alpha}|\p{Digit}|\.|-|_|:|)*';
  my @tags = ();

  my $i = 0;
  while ($norm_text =~ /(<($TAG_NAME_RX)[^>]*\/?>)|(<!($TAG_NAME_RX)[^>]*[^\/]>)|(<\/($TAG_NAME_RX)[^>]*>)|(<!--)|(-->)/g) {
    push @tags, $&;
    $norm_text =~ s/\Q$&\E/" MTEVALXMLTAG".$i." "/e;
    $i++;
        
    }

  # replace entities
  $norm_text =~ s/&quot;/\"/g;  # quote to "
  $norm_text =~ s/&amp;/&/g;   # ampersand to &
  $norm_text =~ s/&lt;/</g;    # less-than to <
  $norm_text =~ s/&gt;/>/g;    # greater-than to >
  $norm_text =~ s/&apos;/\'/g; # apostrophe to '

  # punctuation: tokenize any punctuation unless followed AND preceded by a digit
  $norm_text =~ s/(\P{N})(\p{P})/$1 $2 /g;
  $norm_text =~ s/(\p{P})(\P{N})/ $1 $2/g;

  # cjk: tokenize any CJK chars
  $norm_text =~ s/(\p{InCJK_Unified_Ideographs})/ $1 /g;

  $norm_text =~ s/(\p{S})/ $1 /g; # tokenize symbols

  for ($i = $#tags; $i >= 0; $i--) {
    $norm_text =~ s/MTEVALXMLTAG$i/$tags[$i]/e;
  }

  $norm_text =~ s/\p{Z}+/ /g; # one space only between words
  $norm_text =~ s/^\p{Z}+//; # no leading space
  $norm_text =~ s/\p{Z}+$//; # no trailing space

  return $norm_text;
}

sub add_to_ref {
    my ($file,$REF) = @_;
    my $s=0;
    if ($file =~ /.gz$/) {
	open(REF,"gzip -dc $file|") or die "Can't read $file";
    } else {
	open(REF,$file) or die "Can't read $file";
    }
    binmode REF, ":utf8";
    while(my $ref = <REF>) {
	chop $ref;
	$ref = &tokenization_international($ref) if $tokenization;
	push @{$$REF[$s++]}, $ref;
    }
    close(REF);
}

my(@CORRECT,@TOTAL,$length_translation,$length_reference);

for(my $n=1;$n<=$order;$n++) { $CORRECT[$n] = $TOTAL[$n] = 0; }

my $s=0;
while(<STDIN>) {
    chop;
    $_ = lc if $lowercase;
    $_ = &tokenization_international($_) if $tokenization;
    my @WORD = split;
    my %REF_NGRAM = ();
    my $length_translation_this_sentence = scalar(@WORD);
    my ($closest_diff,$closest_length) = (9999,9999);
    foreach my $reference (@{$REF[$s]}) {
#   print "$s $_ <=> $reference\n";
    $reference = lc($reference) if $lowercase;
	my @WORD = split(' ',$reference);
	my $length = scalar(@WORD);
        my $diff = abs($length_translation_this_sentence-$length);
	if ($diff < $closest_diff) {
	    $closest_diff = $diff;
	    $closest_length = $length;
	    # print STDERR "$s: closest diff ".abs($length_translation_this_sentence-$length)." = abs($length_translation_this_sentence-$length), setting len: $closest_length\n";
	} elsif ($diff == $closest_diff) {
            $closest_length = $length if $length < $closest_length;
            # from two references with the same closeness to me
            # take the *shorter* into account, not the "first" one.
        }
	for(my $n=1;$n<=$order;$n++) {
	    my %REF_NGRAM_N = ();
	    for(my $start=0;$start<=$#WORD-($n-1);$start++) {
		my $ngram = "$n";
		for(my $w=0;$w<$n;$w++) {
		    $ngram .= " ".$WORD[$start+$w];
		}
		$REF_NGRAM_N{$ngram}++;
	    }
	    foreach my $ngram (keys %REF_NGRAM_N) {
		if (!defined($REF_NGRAM{$ngram}) ||
		    $REF_NGRAM{$ngram} < $REF_NGRAM_N{$ngram}) {
		    $REF_NGRAM{$ngram} = $REF_NGRAM_N{$ngram};
#	    print "$i: REF_NGRAM{$ngram} = $REF_NGRAM{$ngram}<BR>\n";
		}
	    }
	}
    }
    $length_translation += $length_translation_this_sentence;
    $length_reference += $closest_length;
    for(my $n=1;$n<=$order;$n++) {
	my %T_NGRAM = ();
	for(my $start=0;$start<=$#WORD-($n-1);$start++) {
	    my $ngram = "$n";
	    for(my $w=0;$w<$n;$w++) {
		$ngram .= " ".$WORD[$start+$w];
	    }
	    $T_NGRAM{$ngram}++;
	}
	foreach my $ngram (keys %T_NGRAM) {
	    $ngram =~ /^(\d+) /;
	    my $n = $1;
            # my $corr = 0;
#	print "$i e $ngram $T_NGRAM{$ngram}<BR>\n";
	    $TOTAL[$n] += $T_NGRAM{$ngram};
	    if (defined($REF_NGRAM{$ngram})) {
		if ($REF_NGRAM{$ngram} >= $T_NGRAM{$ngram}) {
		    $CORRECT[$n] += $T_NGRAM{$ngram};
                    # $corr =  $T_NGRAM{$ngram};
#	    print "$i e correct1 $T_NGRAM{$ngram}<BR>\n";
		}
		else {
		    $CORRECT[$n] += $REF_NGRAM{$ngram};
                    # $corr =  $REF_NGRAM{$ngram};
#	    print "$i e correct2 $REF_NGRAM{$ngram}<BR>\n";
		}
	    }
            # $REF_NGRAM{$ngram} = 0 if !defined $REF_NGRAM{$ngram};
            # print STDERR "$ngram: {$s, $REF_NGRAM{$ngram}, $T_NGRAM{$ngram}, $corr}\n"
	}
    }
    $s++;
}
my $brevity_penalty = 1;
my $bleu = 0;

my @bleu=();

for(my $n=1;$n<=$order;$n++) {
  if (defined ($TOTAL[$n])){
    $bleu[$n]=($TOTAL[$n])?$CORRECT[$n]/$TOTAL[$n]:0;
  }else{
    $bleu[$n]=0;
  }
}

if ($length_reference==0){
  printf "BLEU = 0, 0/0/0/0 (BP=0, ratio=0, hyp_len=0, ref_len=0)\n";
  exit(1);
}

if ($length_translation == 0) {
  $brevity_penalty = 0.0
} elsif ($length_translation<$length_reference) {
  $brevity_penalty = exp(1-$length_reference/$length_translation);
}

my $sumPrec = 0;
for(my $n=1;$n<=$order;$n++) { $sumPrec += my_log( $bleu[$n] ); }
$bleu = $brevity_penalty * exp( $sumPrec ) ;

printf "%f", $bleu;
my $strPrec = "";

for(my $n=1;$n<=$order;$n++) {
    $strPrec .= sprintf "%.1f", 100*$bleu[$n];
    $strPrec .= sprintf "/" if $n < $order;
}

printf STDERR "BLEU = %.2f, %s (BP=%.3f, ratio=%.3f, hyp_len=%d, ref_len=%d)\n",
    100*$bleu,
    $strPrec,
    $brevity_penalty,
    $length_translation / $length_reference,
    $length_translation,
    $length_reference;

sub my_log {
  return -9999999999 unless $_[0];
  return log($_[0]);
}
