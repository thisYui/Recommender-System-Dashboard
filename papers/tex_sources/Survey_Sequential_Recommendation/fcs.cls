% !TeX root = ./template.tex
% !TeX program = pdflatex
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Copyright (C) 2020 by Zhao Chongbin <zhao_chongbin@126.com>
%%% "fcs.cls" is written for Front. Comput. Sci journal template by LaTeX3
%%% syntax. The minimum supported version of package "expl3" is 2017/01/01.
%%% On the other hand, it only supports "pdfTeX" engine. You must use your
%%% typesetting engine to "pdflatex". It provides interfaces as follows:
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Command:
%%% 1. \title{<long title>}
%%% 2. \shorttitle{<short title>}
%%% 3. \author[<address ref clist>]{<author name>}
%%% 4. \author*[<address ref clist>]{<correspond author name>}
%%% 5. \address[<address label>]{<address>}
%%% 6. \fcssetup{<key-value>}
%%%      <key>          : <value>                   |<default>
%%%      doi            = <doi>                     |10.1007/sxxxxx-yyy-zzzz-1
%%%      received       = <received time>           |\today
%%%      accepted       = <accepted time>           |\today
%%%      corr-email     = <correspond author email> |zhao_chongbin@126.com
%%%      issue          = <issue No.>               |0
%%%      volume         = <volume No.>              |0
%%%      first-page     = <first page No.>          |1
%%%      head-year      = <head year>               |\year
%%%      copyright-year = <copyright year>          |\year
%%% 7. \keywords{<keywords>}
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Environment:
%%% 1. abstract
%%% \begin{abstract}
%%%   <abstract context>
%%% \end{abstract}
%%% 2. acknowledgement
%%% \begin{acknowledgement}
%%%   <acknowledgement context>
%%% \end{acknowledgement}
%%% 3. competing interests
%%% \begin{competinginterest}
%%%   <competing interest context>
%%% \end{competinginterest}
%%% 4. biography
%%% \begin{biography}{<author photo file name>}
%%%   <author biography context>
%%% \end{biography}
%%% 5. lemma, definition, corollary, proposition, theorem, example & proof
%%% \begin{<theorem name>}
%%%   <theorem context>
%%% \end{<theorem name>}
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Option:
%%% 1. research(default), review, editorials, letters, perspectives
%%%   "research" describing original investigations relevant to Computer Science;
%%%   "review" giving an in-depth overview of certain topic or a review of one's
%%%     own work or one laboratory or a group of researchers;
%%%   "editorials" providing overview on certain topic written by well known
%%%     scientists in the field.
%%%   example: \documentclass[research]{fcs}
%%% 2. 12pt(default), 10pt
%%%   These are origenal options of standard document class "article.cls".
%%%   Author should use "10pt" to submit.
%%%   Editor should use "12pt" to proofread.
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% More detailed instructions are given below, following the definitions.
%%% Please direct any problem reports to zhao_chongbin@126.com.
%%% This template compatible to Overleaf TeXLive 2017.
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\RequirePackage{expl3}
\ProvidesExplClass{fcs}{2020/09/08}{v1.3}{zsben}
\RequirePackage { xparse, xtemplate, l3keys2e }
\@ifpackagelater { expl3 } { 2019/11/20 }
  {%T
    \cs_set_eq:NN \__fcs_uppercase:n \text_uppercase:n
    \cs_set_eq:NN \__fcs_titlecase:n \text_titlecase:n % 20191120,20200224
  }
  {%F
    \cs_set_eq:NN \__fcs_uppercase:n \MakeUppercase
    \cs_new_protected:Npn \__fcs_titlecase:n #1
      {
        \tl_set:Nn \l__fcs_titlecase_tl { #1 }
        \tl_set:Nn \l__fcs_titlecase_head_tl { \tl_head:n { #1 } }
        \cs_generate_variant:Nn \tl_remove_once:Nn { Nx } % 20171128
        \tl_remove_once:Nx \l__fcs_titlecase_tl \l__fcs_titlecase_head_tl
        \__fcs_uppercase:n \l__fcs_titlecase_head_tl
        \l__fcs_titlecase_tl
      }
  }
\cs_new:Npn \__fcs_msg_new:nn   { \msg_new:nnn { fcs } }
\cs_new:Npn \__fcs_msg_error:nn { \msg_error:nnn { fcs } }
\cs_generate_variant:Nn \__fcs_msg_error:nn { nx } % 20171128
\__fcs_msg_new:nn { l3-too-old }
  {
    Package ~ "#1" ~ is ~ too ~ old. \\\\
    Please ~ update ~ an ~ up-to-date ~ version ~ of ~ the ~ bundles \\
    "l3kernel" ~ and ~ "l3packages" ~ using ~ your ~ TeX ~ package \\
    manager ~ or ~ from ~ CTAN.
  }
\clist_map_inline:nn { expl3, xparse, xtemplate, l3keys2e }
  {
    \@ifpackagelater { #1 } { 2017/01/01 } { }
      { \__fcs_msg_error:nn { l3-too-old } { #1 } }
  }
\__fcs_msg_new:nn { unsupported-engine }
  {
    The~ fcs~ class~ requires~ either~ pdfTeX \\
    "#1"~ is~ not~ supported~ at~ present.~ You~ must~ change \\
    your~ typesetting~ engine~ to~ "pdflatex".
  }
\sys_if_engine_pdftex:F
  {
    \__fcs_msg_error:nx { unsupported-engine } { \c_sys_engine_str }
  }

\tl_new:N \g__fcs_article_type_tl
\tl_new:N \g__fcs_article_fontsize_tl
\cs_new_protected:Npn \__fcs_set_default_value:Nn #1#2
  { \tl_if_empty:NT #1 { \tl_gset:Nn #1 { #2 } } }
\keys_define:nn { fcs / option } % 20171114
  {
    review      .code:n = { \tl_gset:Nn \g__fcs_article_type_tl { review ~ article } },
    research    .code:n = { \tl_gset:Nn \g__fcs_article_type_tl { research ~ article } },
    editorials  .code:n = { \tl_gset:Nn \g__fcs_article_type_tl { editorials } },
    letters     .code:n = { \tl_gset:Nn \g__fcs_article_type_tl { letters } },
    perspectives.code:n = { \tl_gset:Nn \g__fcs_article_type_tl { perspectives } },
    10pt        .code:n = { \tl_gset:Nn \g__fcs_article_fontsize_tl { 10pt } },
    12pt        .code:n = { \tl_gset:Nn \g__fcs_article_fontsize_tl { 12pt } },
    review      .value_forbidden:n = true,
    research    .value_forbidden:n = true,
    editorials  .value_forbidden:n = true,
    letters     .value_forbidden:n = true,
    perspectives.value_forbidden:n = true,
    10pt        .value_forbidden:n = true,
    12pt        .value_forbidden:n = true,
  }
\ProcessKeysOptions { fcs / option }
\__fcs_set_default_value:Nn \g__fcs_article_type_tl { research ~ article }
\__fcs_set_default_value:Nn \g__fcs_article_fontsize_tl { 12pt }
% \DeclareOption* { \PassOptionsToClass { \CurrentOption } { article } }
% \ProcessOptions \relax
\LoadClass [ twoside, \g__fcs_article_fontsize_tl ] { article }

\RequirePackage
  {
    amsmath, amssymb, amsthm, geometry, fancyhdr, lastpage,
    graphicx, caption, titlesec, txfonts, xcolor, %balance,
    marvosym, booktabs, cite, etoolbox, wrapfig, enumitem,
    subfigure, stfloats, array, tabularx, flushend, %showframe,
  }
\RequirePackage [ colorlinks ] { hyperref }

\graphicspath { { fig/ } }

\geometry{
  paperwidth = 210truemm,
  paperheight = 277truemm,
  textwidth = 176truemm,
  textheight = 233truemm,
  top = 9mm,
  % bottom = 18mm,
  headsep = 6mm,
  footskip = 11mm,
  headheight = 6mm,
  hcentering,
  includeheadfoot,
}

\RenewExpandableDocumentCommand \baselinestretch { } { 1.2 }
\skip_set:Nn \baselineskip { 14pt plus 5pt minus 3pt  }
% \skip_set:Nn \lineskip { 5pt plus 5pt minus 3pt }
% \dim_set:Nn \lineskiplimit { 5pt }
\dim_set:Nn \columnsep { 5mm }
\dim_set:Nn \parindent { 4mm }

% \tl_set:Nn \UrlBreaks
%   {
%     \do\a\do\b\do\c\do\d\do\e\do\f\do\g\do\h\do\i\do\j\do\k\do\l\do\m
%     \do\n\do\o\do\p\do\q\do\r\do\s\do\t\do\u\do\v\do\w\do\x\do\y\do\z
%     \do\A\do\B\do\C\do\D\do\E\do\F\do\G\do\H\do\I\do\J\do\K\do\L\do\M
%     \do\N\do\O\do\P\do\Q\do\R\do\S\do\T\do\U\do\V\do\W\do\X\do\Y\do\Z
%     \do0\do1\do2\do3\do4\do5\do6\do7\do8\do9
%   }

\NewDocumentCommand \fcssetup { m }
  { \keys_set:nn { fcs } { #1 } }
\keys_define:nn { fcs }
  {
    doi            .tl_set:N  = \l__fcs_doi_tl,
    received       .tl_set:N  = \l__fcs_received_time_tl,
    accepted       .tl_set:N  = \l__fcs_accepted_time_tl,
    corr-email     .tl_set:N  = \l__fcs_corr_email_tl,
    issue          .int_set:N = \l__fcs_issue_int,
    volume         .int_set:N = \l__fcs_volume_int,
    first-page     .int_set:N = \l__fcs_first_page_int,
    head-year      .int_set:N = \l__fcs_head_year_int,
    copyright-year .int_set:N = \l__fcs_copyright_year_int,

    doi            .initial:n = { 10.1007/sxxxxx-yyy-zzzz-1 },
    received       .initial:n = \today,
    accepted       .initial:n = \today,
    corr-email     .initial:n = { zhao_chongbin@126.com },
    issue          .initial:n = { 0 },
    volume         .initial:n = { 0 },
    first-page     .initial:n = { 1 },
    head-year      .initial:n = \year,
    copyright-year .initial:n = \year,

    first-page     .code:n    =
      {
        \int_set:cn { c@page } { #1 }
        \int_set:Nn \l__fcs_first_page_int { #1 }
      },
  }

\fancyhf { }
\fancyhead [ RO, LE ] { \small \int_use:c { c@page } }
\fancyhead [ CO ] { \small \g__fcs_short_author_tl .~ \g__fcs_short_title_tl }
\fancyhead [ CE ] { \small \l__fcs_year_issue_volume_page_tl }
\pagestyle { fancy }
\fancypagestyle { plain }
  {
    \skip_add:Nn \headheight \baselineskip
    \fancyhf { }
    \fancyhead [ LO, RE ]
      {
        \small
        \l__fcs_year_issue_volume_page_tl \\[1mm]
        \l__fcs_doi_url_tl
      }
    % \fancyhead [ RO, LE ] { \small \int_use:c { c@page } \\ }
  }

\RenewDocumentCommand \headrulewidth { } { 0pt }

\tl_new:N \g__fcs_first_author_tl
\tl_new:N \g__fcs_short_author_tl
\tl_new:N \g__fcs_short_title_tl
\NewDocumentCommand \shorttitle { m }
  {
    \tl_gset:Nn \g__fcs_short_title_tl { #1 }
  }

\DeclareDocumentCommand \title { m }
  {
    \tl_set:Nn \l__fcs_title_tl
      {
        \group_begin:
          \LARGE \bfseries #1 \par
        \group_end:
      }
    \__fcs_set_default_value:Nn \g__fcs_short_title_tl { #1 }
  }

\seq_new:N \g__fcs_authors_seq
\seq_new:N \g__fcs_addresses_seq

\DeclareDocumentCommand \author { s o m }
  {
    \IfNoValueTF { #2 }
      {
        \seq_put_right:Nn \g__fcs_authors_seq
          {
            #3
            \IfBooleanT { #1 } { \l__fcs_correspond_author_mark_tl }
          }
      }
      {
        \seq_put_right:Nn \g__fcs_authors_seq
          {
            #3
            \IfBooleanT { #1 } { \l__fcs_correspond_author_mark_tl }
            \clist_set:Nx \l_tmpa_clist
              { \clist_map_function:nN { #2 } \__fcs_ref:n }
            \textsuperscript { \clist_use:Nn \l_tmpa_clist { , } }
          }
      }
    \__fcs_set_default_value:Nn \g__fcs_first_author_tl { #3 }
  }
\cs_new:Npn \__fcs_ref:n #1
  { \exp_not:N \ref { #1 } , }

\DeclareDocumentCommand \address { o m }
  {
    \IfNoValueTF { #1 }
      { \seq_put_right:Nn \g__fcs_addresses_seq { #2 } }
      {
        \seq_put_right:Nn \g__fcs_addresses_seq
          {
            \refstepcounter { fcs@address }
            \label { #1 }
            \int_use:c { c@fcs@address }
            \quad #2
          }
      }
  }
\newcounter { fcs@address }
\setcounter { fcs@address } { 0 }

\cs_new_protected:Npn \__fcs_abstract_keywords_format:nn #1#2
  {
    \noindent \bfseries #1 \quad \normalfont #2 \par
  }

\@ifpackagelater { xparse } { 2019/01/01 }
  {%T
    \DeclareDocumentEnvironment { abstract } { +b }
      {
        \tl_gset:Nn \g__fcs_abstract_tl
          { \__fcs_abstract_keywords_format:nn { Abstract } { #1 } }
      } { }

    \NewDocumentEnvironment { biography } { m +b }
      {
        % \__fcs_biography:nn { #1 } { #2 }
        \__fcs_biography_surround:nn { #1 } { #2 }
      } { }
  }
  {%F
    \RequirePackage { environ }
    \RenewEnviron { abstract }
      {
        \tl_gset:Nx \g__fcs_abstract_tl
          { \__fcs_abstract_keywords_format:nn { Abstract } { \BODY } }
      }

    \NewEnviron { biography } [ 1 ]
      {
        % \__fcs_biography:nn { #1 } { \BODY }
        \__fcs_biography_surround:nn { #1 } { \BODY }
      }
  }

\DeclareDocumentCommand \keywords { m }
  {
    \tl_set:Nn \l__fcs_keywords_tl
      { \__fcs_abstract_keywords_format:nn { Keywords } { #1 } }
  }

\tl_set:Nn \l__fcs_hrule_height_tl { .5pt }
\cs_set:Nn \__fcs_hrulefill:
  {
    \noindent
    % \hrulefill
    \rule [ -\l__fcs_hrule_height_tl ] { \linewidth } { \l__fcs_hrule_height_tl }
  }
\box_new:N \l__fcs_article_type_hbox
\hbox_set:Nn \l__fcs_article_type_hbox
  {
    \hbox_to_wd:nn { 0pt }
      { \rule [ -1pt ] { \textwidth } { 1pt } }
    \minipage [ t ] { .5\textwidth }
      \skip_vertical:N 0pt
      \colorbox { black }
        {
          \hbox_to_wd:nn \textwidth
            {
              \skip_horizontal:N 1.5ex
              \vbox_to_ht:nn { 10pt }
                {
                  \vfil
                  \color { white }
                  % \bfseries
                  \__fcs_uppercase:n \g__fcs_article_type_tl
                  \vfil
                }
              \hfil
            }
        }
    \endminipage
  }
\DeclareDocumentCommand \maketitle { }
  {
    \thispagestyle { plain }
    \twocolumn
      [
        \centering
        \skip_vertical:N -.7em
        \box_use:N \l__fcs_article_type_hbox
        \skip_vertical:N 39pt
        \l__fcs_title_tl
        \skip_vertical:N 11mm \bfseries \large
        % \seq_use:Nnnn \g__fcs_authors_seq { ~ and ~ } { ,~ } { ~ and ~ }
        \seq_use:Nn \g__fcs_authors_seq { ,~ }

        \if_int_compare:w 1 < \seq_count:N \g__fcs_authors_seq
          \tl_gset:Nn \g__fcs_short_author_tl { \g__fcs_first_author_tl \ \l__fcs_etal_tl}
        \else:
          \tl_gset:Nn \g__fcs_short_author_tl \g__fcs_first_author_tl
        \fi:

        \skip_vertical:N 4mm \normalfont \normalsize%\small
        \seq_use:Nn \g__fcs_addresses_seq { \par }
        \skip_vertical:N 9.5mm \raggedright \small
        \l__fcs_copyright_tl
        \skip_vertical:N 12mm
      ]
    \g__fcs_abstract_tl
    \skip_vertical:N 1ex
    \l__fcs_keywords_tl
    % \skip_vertical:N 1ex
    \RenewExpandableDocumentCommand \footnoterule { } { \__fcs_hrulefill: }
    \RenewExpandableDocumentCommand \thefootnote { } { }
    \footnotetext
      {
        \skip_vertical:N -1.5ex
        \noindent
        \seq_use:Nn \g__fcs_thanks_seq
          {
            \skip_vertical:N -1.5ex
            \__fcs_hrulefill: \\
          }
      }
    \RenewExpandableDocumentCommand \thefootnote { }
      {
        \int_use:c { c@footnote } )
      }
  }


\setcounter { secnumdepth } { 4 }
\titleformat
  { \section }
  {
    \RenewExpandableDocumentCommand \baselinestretch { } { 1 }
    \titlerule [ \l__fcs_hrule_height_tl ]
    \skip_vertical:N 2mm plus 1mm
    \normalfont \large \bfseries
  }
  { \thesection }
  { 1em }
  { }
\titleformat
  { \subsection }
  {
    \RenewExpandableDocumentCommand \baselinestretch { } { 1 }
    \normalfont \normalsize
  }
  { \thesubsection }
  { 1em }
  { }
\titleformat
  { \subsubsection }
  {
    \RenewExpandableDocumentCommand \baselinestretch { } { 1 }
    \normalfont \normalsize
  }
  { \thesubsubsection }
  { 1em }
  { }
\titleformat
  { \paragraph }
  { \normalfont \normalsize }
  { $\bullet$ }
  { 1ex }
  { }
\titleformat
  { \subparagraph }
  [ runin ]
  { \normalfont \normalsize \bfseries }
  { }
  { 0pt }
  { }

\titlespacing*
  { \section }
  { 0pt }
  { 6mm plus 4mm minus 2mm }
  { 2mm plus 2mm minus 1mm }
% \titlespacing*
%   { \subsection }
%   { 0pt }
%   { 3mm }
%   { 3mm }
% \titlespacing*
%   { \subsubsection }
%   { 0pt }
%   { .5ex plus .5ex minus .5ex }
%   { .5ex plus .5ex minus .5ex }
\titlespacing*
  { \paragraph }
  { 0pt }
  { 1pt plus 2pt minus 1pt}
  { 1pt plus 2pt minus 1pt }
\titlespacing
  { \subparagraph }
  { \parindent }
  { 0pt plus 1pt }
  { 1em }

\setlist{
  itemsep = 0pt plus 2pt,
  parsep  = 0pt plus 2pt,
  topsep  = 0pt plus 2pt,
}

\captionsetup
  {
    font      = footnotesize,
    labelfont = bf,
    labelsep  = quad,
    belowskip = 0pt,
  }
\captionsetup [ table ]
  {
    justification   = RaggedRight,
    singlelinecheck = false,
    position        = top,
    aboveskip       = 0pt plus 3pt
  }
\captionsetup [ figure ]
  {
    aboveskip = 5pt plus 3pt minus 3pt,
    % margin = 8mm
  }
\skip_set:Nn \intextsep { 4mm }
% \skip_set:Nn \tabcolsep { 0pt }
\cs_set_eq:NN \__fcs_tabularx:N \tabularx
\RenewDocumentCommand \tabularx { } { \__fcs_tabularx:N \linewidth }
\RenewDocumentCommand \figurename { } { Fig. }
\BeforeBeginEnvironment { tabular } { \footnotesize }
\BeforeBeginEnvironment { tabular* } { \footnotesize }
\BeforeBeginEnvironment { tabularx } { \footnotesize }
\newcolumntype { L } { X }
\newcolumntype { R } { >{\raggedleft\arraybackslash} X }
\newcolumntype { C } { >{\centering\arraybackslash}  X }

\theoremstyle { definition }
\cs_new_protected_nopar:Npn \__fcs_new_theorem:n #1
  {
    \newtheorem { #1 } { \__fcs_titlecase:n { #1 } }
    \newtheorem* { #1* } { \__fcs_titlecase:n { #1 } }
  }
\clist_set:Nn \l__theorem_label_clist
  {
    lemma, definition, corollary, proposition, theorem, example
  }
\clist_map_function:NN \l__theorem_label_clist \__fcs_new_theorem:n

\NewDocumentEnvironment { acknowledgement } { }
  {
    \skip_vertical:N 4mm
    \noindent
    \dim_set:Nn \baselineskip { 10.5pt }
    \RenewExpandableDocumentCommand \baselinestretch { } { 1.05 }
    \group_begin:
      \footnotesize
      \group_begin:
        \bfseries Acknowledgements
      \group_end:
      \quad
  }
  {
    \group_end:
    \par
    \skip_set:Nn \baselineskip { 15pt plus 5pt minus 3pt }
    \RenewExpandableDocumentCommand \baselinestretch { } { 1 }
  }

\NewDocumentEnvironment { competinginterest } { }
  {
    \skip_vertical:N 4mm
    \noindent
    \dim_set:Nn \baselineskip { 10.5pt }
    \RenewExpandableDocumentCommand \baselinestretch { } { 1.05 }
    \group_begin:
      \footnotesize
      \group_begin:
        \bfseries Competing~ interests
      \group_end:
      \quad
  }
  {
    \group_end:
    \par
    \skip_set:Nn \baselineskip { 15pt plus 5pt minus 3pt }
    \RenewExpandableDocumentCommand \baselinestretch { } { 1 }
  }

\RenewDocumentEnvironment { thebibliography } { m }
  {
    % \balance
    \section* { \refname }
    \RenewExpandableDocumentCommand \baselinestretch { } { 1 }
    \footnotesize
    \dim_set:Nn \baselineskip { 12.8pt }% plus 1pt minus 1pt
    \list
      {
        \dim_gset:Nn \labelsep { 8pt }
        \dim_gset:Nn \topsep { 10bp }
        \dim_gset:Nn \partopsep { 5pt }
        \dim_gset:Nn \parskip { 0pt }
        \dim_gset:Nn \itemsep { 0pt }
        \dim_gset:Nn \leftmargin { \labelwidth + \labelsep }
        \int_use:c { c@enumiv }.
      }
      {
        \settowidth \labelwidth { \@biblabel { #1 } }
        \leftmargin \labelwidth
        \dim_add:Nn \leftmargin \labelsep
        \@openbib@code
        \usecounter { enumiv }
        \cs_set_eq:cc { p@enumiv } { @empty }
        \RenewDocumentCommand \theenumiv { } { \int_use:c { c@enumiv } }
      }
    \sloppy
    \tex_clubpenalty:D 4000
    \@clubpenalty \tex_clubpenalty:D
    \tex_widowpenalty:D 4000
    \char_set_sfcode:nn { `\. } \@m
  }
  {
    \cs_set_nopar:Npn \@noitemerr
      { \msg_warning:nn { fcs } { Empty~"thebibliography"~environment } }
    \endlist
  }

\box_new:N \l__fcs_biography_photo_hbox
\box_new:N \l__fcs_biography_text_vbox
\dim_new:N \l__fcs_biography_photo_ht_dp_dim
\dim_new:N \l__fcs_biography_text_ht_dp_dim
\int_new:N \l__fcs_biography_photo_line_num_int
\cs_new_protected:Npn \box_ht_dp:N #1
  { \box_ht:N #1 + \box_dp:N #1 }

% #1: figure name, #2: context
\cs_new_protected:Npn \__fcs_biography_surround:nn #1#2
  {
    \skip_vertical:N 2em plus 1em minus .5em
    \noindent
    \minipage { \linewidth }
      \dim_set:Nn \parskip { 0mm }
      \dim_set:Nn \parindent { 1em }
      \skip_set:Nn \baselineskip { 15pt plus 5pt minus 3pt }
      \RenewExpandableDocumentCommand \baselinestretch { } { 1.25 }
      \hbox_set:Nn \l__fcs_M_box { \small M }
      \dim_set:Nn \l__fcs_M_ht_dim { \box_ht:N \l__fcs_M_box }

      \hbox_set:Nn \l__fcs_biography_photo_hbox % 20170405
        { \includegraphics [ width = \c__fcs_biography_photo_width_dim ] { #1 } }
      \dim_set:Nn \l__fcs_biography_photo_ht_dp_dim
        { \box_ht_dp:N \l__fcs_biography_photo_hbox }
      \int_set:Nn \l__fcs_biography_photo_line_num_int % Number of lines occupied by picture
        {
          \fp_to_int:n
            {
              \fp_eval:n
                { ceil ( \l__fcs_biography_photo_ht_dp_dim / \baselineskip ) }
            }
        }

      \begin { wrapfigure } [ \l__fcs_biography_photo_line_num_int ] { l } { 2.8cm }
        \tl_if_eq:NnTF \g__fcs_article_fontsize_tl { 10pt }
          { \skip_vertical:N -\baselineskip }
          { \skip_vertical:N -.7\baselineskip }
        \box_use:N \l__fcs_biography_photo_hbox
      \end { wrapfigure }
      \small \noindent

      \vbox_set:Nn \l__fcs_biography_text_vbox { #2 } % 20170405
      \dim_set:Nn \l__fcs_biography_text_ht_dp_dim
        { \box_ht_dp:N \l__fcs_biography_text_vbox }

      \if_dim:w \l__fcs_biography_text_ht_dp_dim < \l__fcs_biography_photo_ht_dp_dim
        \parbox [ t ] [ \l__fcs_biography_photo_ht_dp_dim ] [ t ] { \linewidth } { #2 }
      \else:
        #2
      \fi:
    \endminipage
  }

\box_new:N \l__fcs_M_box
\dim_new:N \l__fcs_M_ht_dim
\dim_const:Nn \c__fcs_biography_photo_width_dim { 2.5cm }
\dim_const:Nn \c__fcs_biography_tabular_space_dim { 2em }
\dim_new:N \l__fcs_biography_text_width_dim
% #1: figure name, #2: context
\cs_new_protected:Npn \__fcs_biography:nn #1#2
  {
    \hbox_set:Nn \l__fcs_M_box { \small M }
    \dim_set:Nn \l__fcs_M_ht_dim { \box_ht:N \l__fcs_M_box }
    \dim_set:Nn \l__fcs_biography_text_width_dim
      { \linewidth - \c__fcs_biography_photo_width_dim - \c__fcs_biography_tabular_space_dim }

    \skip_vertical:N 2em plus 1em minus .5em
    \noindent
    \tabular
      {
        @ { }
        p { \c__fcs_biography_photo_width_dim }
        @ { }
        p { \c__fcs_biography_tabular_space_dim }
        @ { }
        p { \l__fcs_biography_text_width_dim }
        @ { }
      }
      \minipage [ t ] { \linewidth }
        \skip_vertical:N -\l__fcs_M_ht_dim
        \includegraphics [ width = \linewidth ] { #1 }
      \endminipage
      &&
      \parbox [ t ] { \linewidth } { \small #2 }
    \endtabular
  }

\tl_set:Nn \l__fcs_etal_tl
  {
    et ~ al
  }
\tl_set:Nn \l__fcs_correspond_author_mark_tl
  {
    ( { \Large \Letter } ) % The control sequence \Letter is from package "marvosym"
  }
\tl_set:Nn \l__fcs_year_issue_volume_page_tl
  {
    \l__fcs_journal_name_tl
    , ~
    \int_use:N \l__fcs_head_year_int
    , ~
    \int_to_arabic:n \l__fcs_issue_int
    (
    \int_to_arabic:n \l__fcs_volume_int
    ): ~
    \int_to_arabic:n \l__fcs_first_page_int
    % \int_use:c { c@page }
    --
    \pageref* { LastPage }
  }
\tl_set:Nn \l__fcs_journal_name_tl
  {
    Front. ~ Comput. ~ Sci.
  }
\tl_set:Nn \l__fcs_doi_url_tl
  {
    https://doi.org/ \l__fcs_doi_tl
  }
\tl_set:Nn \l__fcs_copyright_tl
  {
    \copyright\
    % Higher ~ Education ~ Press ~ and ~ Springer-Verlag ~ Berlin ~ Heidelberg ~
    % Higher ~ Education ~ Press ~ and ~ Springer-Verlag ~ GmbH ~ Germany, ~ part ~ of ~ Springer ~ Nature ~
    Higher ~ Education ~ Press ~
    \int_use:N \l__fcs_copyright_year_int
  }

\seq_new:N \g__fcs_thanks_seq
\cs_new_protected:Npn \__fcs_put_thanks:n
  {
    \seq_put_right:Nn \g__fcs_thanks_seq
  }
\AfterPreamble
  {
    \__fcs_put_thanks:n
      { Received ~ \l__fcs_received_time_tl; ~ accepted ~ \l__fcs_accepted_time_tl }
    \__fcs_put_thanks:n { E-mail: ~ \l__fcs_corr_email_tl }
    % \RequirePackage [ colorlinks ] { hyperref }
  }
\AfterEndPreamble
  {
    \maketitle
  }
