jQuery(document).ready(function () {
    $('#btn-process').on('click', function () {
        sent_1 = $('#sentence-1').val();
        sent_2 = $('#sentence-2').val();
        model = $('#input_model').val();
        $.ajax({
            url: '/predict',
            type: "post",
            contentType: "application/json",
            dataType: "json",
            data: JSON.stringify({
                "sent_1": sent_1,
                "sent_2": sent_2,
                "model": model
            }),
        }).done(function (jsondata, textStatus, jqXHR) {
            result = "";
            output = jsondata['output'];
            if(output == 1){
                result = "Yes";
            }
            else{
                result = "No";
            }
            document.getElementById("output-txt").innerHTML = model.toUpperCase()+" says: "+result;
        }).fail(function (jsondata, textStatus, jqXHR) {
            alert(jsondata['responseJSON']['message'])
        });
    })
})