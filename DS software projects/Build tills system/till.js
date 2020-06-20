var setNum;
var setTotal;
var setId;
var setShiftId;
var setShiftLabel;
var setReset;
var setRefund;

var pos_xmlhttp;

function sushi_random()
{
    var rn;

    rn = Math.floor(Math.random()*100001);
    return '&randn='+rn;
}


function requestObject()
{
  var xmlhttp;
  if (window.XMLHttpRequest)
  {
    // code for IE7+, Firefox, Chrome, Opera, Safari
    xmlhttp=new XMLHttpRequest();
  } else
  {
    // code for IE6, IE5
    xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
  }

  return xmlhttp;
}

function sushi_process_common(type, aa)
{
    if (type == 'redirect')
    {
      var where = aa.getElementsByTagName('url')[0].childNodes[0].nodeValue;
      where = where.replace(/\%26/g,'&');
      window.location = where;

    } else if (type == 'refill')
    {
      var where = aa.getElementsByTagName('where')[0].childNodes[0].nodeValue;
      var what  = aa.getElementsByTagName('what')[0].childNodes[0].nodeValue;
      var where_html = document.getElementById(where);

      if (where_html != null)
      {
        where_html.innerHTML = sushi64(what);
      }

    } else if (type == 'append')
    {
      var where = aa.getElementsByTagName('where')[0].childNodes[0].nodeValue;
      var what  = aa.getElementsByTagName('what')[0].childNodes[0].nodeValue;
      var where_html = document.getElementById(where);

      if (where_html != null)
      {
        where_html.innerHTML = where_html.innerHTML + sushi64(what);
      }

    }
}

function sushi64(input)
{
  var key = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=";
  var output = "";
  var chr1, chr2, chr3, enc1, enc2, enc3, enc4;
  var i = 0;

  input = input.replace(/[^A-Za-z0-9\+\/\=]/g, "");

  while (i < input.length)
  {
    enc1 = key.indexOf(input.charAt(i++));
    enc2 = key.indexOf(input.charAt(i++));
    enc3 = key.indexOf(input.charAt(i++));
    enc4 = key.indexOf(input.charAt(i++));

    chr1 =  (enc1 << 2)       | (enc2 >> 4);
    chr2 = ((enc2 & 15) << 4) | (enc3 >> 2);
    chr3 = ((enc3 & 3) << 6)  |  enc4;

    output = output + String.fromCharCode(chr1);

    if (enc3 != 64)
    {
      output = output + String.fromCharCode(chr2);
    }

    if (enc4 != 64)
    {
      output = output + String.fromCharCode(chr3);
    }
  }

  return output;//utf8_decode(output);
}



function sushi64p(input)
{
  var key = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=";
  var output = [];
  var chr1, chr2, chr3, enc1, enc2, enc3, enc4;
  var i = 0;
  var pos = 0;

  var el,al;

  input = input.replace(/[^A-Za-z0-9\+\/\=]/g, "");

  el = 0;
 
  while (i < input.length)
  {
    enc1 = key.indexOf(input.charAt(i++));
    enc2 = key.indexOf(input.charAt(i++));
    enc3 = key.indexOf(input.charAt(i++));
    enc4 = key.indexOf(input.charAt(i++));
 
    chr1 =  (enc1 << 2)       | (enc2 >> 4);
    chr2 = ((enc2 & 15) << 4) | ((enc3 & 15)>> 2);
    chr3 = ((enc3 & 3) << 6)  |  (enc4 & 63);
 
    output[pos] = chr1; pos += 1;
    el += 1;
 
    if (enc3 != 64)
    {
      output[pos] = chr2; pos += 1;
      el += 1;
    }

    if (enc4 != 64)
    {
      output[pos] = chr3; pos += 1;
      el += 1;
    }
  }
  
  ml = output.length;

  return output;
}


function pos_response()
{
  var content;
  var res;
  var e;

  if (pos_xmlhttp.readyState == 4)
  {
    res = pos_xmlhttp.responseXML;



    res.normalize();

    pos_process_extend(res);

    e = document.getElementById('ins');


    if (e != null)
    {
      pos_local_clr();
      pos_refresh();
    }
  }
}

function pos_response_norefresh()
{
  var content;
  var res;
  var e;

  if (pos_xmlhttp.readyState == 4)
  {
    res = pos_xmlhttp.responseXML;

    res.normalize();

    pos_process_extend(res);

    e = document.getElementById('ins');


    if (e != null)
    {
      pos_local_clr();
    }
  }
}

function pos_process_extend(response)
{
  var aa;

  aa = response.getElementsByTagName('action');

  for(var i = 0, len = aa.length; i < len; i += 1)
  {
    var type = aa[i].getElementsByTagName('type')[0].childNodes[0].nodeValue;

    if (type == 'total')
    {
      var total = aa[i].getElementsByTagName('value')[0].childNodes[0].nodeValue;
      setTotal = total;

    } else if (type == 'highlight')
    {
      var id = aa[i].getElementsByTagName('iid')[0].childNodes[0].nodeValue;
      var e = document.getElementById(id);
      e.style.background='#ff0';

    } else if (type == 'lowlight')
    {
      var id = aa[i].getElementsByTagName('iid')[0].childNodes[0].nodeValue;
      var e = document.getElementById(id);
      e.style.background='#eee';

    } else if (type == 'reset')
    {
	setReset = 1;

    } else
    {
      sushi_process_common(type, aa[i]);
    }
  }
}

function pos_refresh()
{
    var e,t,s;
    var x, h;
    s = document.getElementById('lstate1');

    h='';

    if (setRefund == 4)
    {
	h='Refund ';
    }

    if (setNum != 1)
    {
	h = h+'x '+setNum;
    }

    if (h == '')
    {
	h = '&nbsp;';
    }

    s.innerHTML = h;

    s = document.getElementById('lstate2');

    if (setShiftId != 0)
    {
      s.innerHTML = setShiftLabel;
    } else
    {
      s.innerHTML = '&nbsp;';
    }


    s = document.getElementById('rstate2');

    if (setTotal < 0)
    {
      s.innerHTML = 'Total -&pound;'+Number(setTotal/-100).toFixed(2);
    } else
    {
      s.innerHTML = 'Total &pound;'+Number(setTotal/100).toFixed(2);
    }

    e = document.getElementById('ins');
    t = document.getElementById('target');
    t.scrollTop = t.scrollHeight;
    e.focus();
}

function pos_page(id)
{
    pos_reset();
    pos_xmlhttp = requestObject();
    pos_xmlhttp.onreadystatechange=pos_response;
    pos_xmlhttp.open('GET','/action?action=page.php&page='+id+sushi_random(),true);
    pos_xmlhttp.send(null);
}

function add_item(refund, num, name, cost)
{
    var e,t,m;
    setTotal = Number(setTotal) + Number(cost * num);
    t = document.getElementById('target');

    m = '<div id="item'+setId+'" style=""><a href="" onclick="highlight(\'item'+setId+'\');return false;">';
    setId = setId + 1;

    if (num != 1)
    {
      if (refund == 4)
      {
        m = m + '<div style="margin-left:6px;">Refund '+setNum+' @ &pound;'+Number(cost).toFixed(2)+'</div>';
      } else
      {
        m = m + '<div style="margin-left:6px;">'+setNum+' @ &pound;'+Number(cost).toFixed(2)+'</div>';
      }

    } else
    {
      if (refund == 4)
      {
	m = m + '<div style="margin-left:6px;">Refund</div>';
      }
    }
	
    m = m + '<div class="psl">'+name+'</div><div class="psr">&pound;'+Number(cost * num).toFixed(2)+'</div>';

    m = m + '<div class="psc"></div></a></div>';
    t.innerHTML = t.innerHTML + m;
}

function pos_start(page)
{
    var e;
    e = document.getElementById('ins');

    e.value = '';
    setFound = 0;
    setTotal = 0;
    setReset = 0;
    pos_local_clr();
    pos_refresh();
    pos_xmlhttp = requestObject();
    pos_xmlhttp.onreadystatechange=pos_response;
    pos_xmlhttp.open('GET','/action?action=status&page='+page+sushi_random(),true);
    pos_xmlhttp.send(null);
}

function pos_reset()
{
    var e;
    if (setReset == 1)
    {
	setReset = 0;
	pos_local_clr();
	setTotal = 0;
	e = document.getElementById('target');
	if (null != e)
        {
	  e.innerHTML = '';
	}

	e = document.getElementById('title');
	if (null != e)
        {
	  e.innerHTML = 'Till System';
	}

	pos_refresh();
    }
}

function pos_digit(argy)
{
    var e;
    pos_reset();
    e = document.getElementById('ins');

    if (argy == 100)
    {
      e.value = e.value + '00';
    } else
    {
    e.value = e.value + argy;
    }

    pos_refresh();
}

function pos_clr()
{
    pos_reset();
    pos_local_clr();

    pos_xmlhttp = requestObject();
    pos_xmlhttp.onreadystatechange=pos_response;
    pos_xmlhttp.open('GET','/action?action=clr'+sushi_random(),true);
    pos_xmlhttp.send(null);
}

function pos_local_clr()
{
    var e;
    e = document.getElementById('ins');
    if (e != null)
    {
      e.value='';
    }
    setRefund = 0;
    setNum = 1;
    setShiftId = 0;
    setShiftLabel = '';
    pos_refresh();
}

function pos_local_clr_not_ref()
{
    var e;
    e = document.getElementById('ins');
    e.value='';
    setNum = 1;
    setShiftId = 0;
    setShiftLabel = '';
}

function pos_plu(arg_p)
{
    var e,v;
    pos_reset();
    pos_xmlhttp = requestObject();
    pos_xmlhttp.onreadystatechange=pos_response;

    e = document.getElementById('ins');
    v = e.value;

    pos_xmlhttp.open("GET","/action?action=plu&refund="+setRefund+"&num="+setNum+'&plu='+v+sushi_random(),true);
    pos_xmlhttp.send(null);
}

function pos_program(arg_pos)
{
    var e,v;
    pos_reset();
    pos_xmlhttp = requestObject();
    pos_xmlhttp.onreadystatechange=pos_response;

    e = document.getElementById('ins');
    v = e.value;

    pos_xmlhttp.open("GET","/action?action=program&refund="+setRefund+"&num="+setNum+'&pos='+arg_pos+'&shift='+setShiftId+'&value='+v+sushi_random(),true);
    pos_xmlhttp.send(null);
}

function pos_plus(arg_p)
{
    var e,v;
    pos_reset();
    pos_xmlhttp = requestObject();
    pos_xmlhttp.onreadystatechange=pos_response;

    e = document.getElementById('ins');
    v = e.value;

    pos_xmlhttp.open("GET","/action?action=plus&refund="+setRefund+"&num="+setNum+'&pid='+arg_p+'&value='+v+'&shift='+setShiftId+sushi_random(),true);
    pos_xmlhttp.send(null);
}


function pos_point(arg_p)
{
    var e;
    pos_reset();
    e = document.getElementById('ins');
    if (e.value=='')
    {
	e.value='0.';
    } else
    {
	e.value = e.value+'.';
    }

    pos_refresh();
}

function pos_tender(n)
{
    var e;

    pos_reset();

    e = document.getElementById('ins');

    if (e.value=='')
    {
	e.value = n;
    } else
    {
	e.value = Number(e.value) + Number(n);
    }

    pos_refresh();
}

function pos_cash(n)
{
    var e,c,r;
    pos_reset();
    e = document.getElementById('ins');
    c = e.value;
    r = setRefund;

    if (c == '')
    {
	if (setTotal >= 0)
	{
	    c = setTotal;
	} else
	{
	    if (r != 0)
	    {
	      c = -setTotal;
	    } else
	    {
		c = 0;
	    }
	}
    }

    pos_local_clr();
    pos_xmlhttp = requestObject();
    pos_xmlhttp.onreadystatechange=pos_response;
    pos_xmlhttp.open('GET','/action?action=cash&refund='+r+'&type='+n+'&cash='+c+sushi_random(),true);
    pos_xmlhttp.send(null);
}

function pos_num()
{
    var e;
    pos_reset();
    e = document.getElementById('ins');
    if (e.value!='')
    {
	setNum = e.value;
    }
    e.value = '';
    pos_refresh();
}

function pos_refund()
{
    pos_reset();
    if (setRefund == 0)
    {
      setRefund = 4;
    } else
    {
      setRefund = 0;
    }
    pos_refresh();
}

function pos_shift(arg_n, arg_m)
{
    pos_reset();
    setShiftId    = arg_n;
    setShiftLabel = arg_m;
    pos_refresh();
}

function pos_total()
{
    var e;
    pos_reset();
    pos_local_clr_not_ref();
    e = document.getElementById('ins');

    if (setTotal < 0)
    {
      e.value = -setTotal;
    } else
    {
      e.value = setTotal;
    }

    pos_refresh();
}

function pos_page(page)
{
	if (page == 1)
	{
		window.location.href = '/';
	} else if (page == 2)
	{
		window.location.href = '/till2.html';
	}
}
