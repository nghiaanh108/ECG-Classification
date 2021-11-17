$(document).ready(function () {
    // Init
    $('.image-section').show();
    $('.loader').hide();
    $('#result').hide();
    $('#score').hide();
    $('#filename').hide();
    $('#time_predict').hide();
    $('#status').hide();
    $('#item').hide();
    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        $('#score').text('');
        $('#score').hide();
        $('#filename').text('');
        $('#filename').hide();
        $('#time_predict').text('');
        $('#time_predict').hide();
        $('#status').text('');
        $('#status').hide();
        $('#item').text('');
        $('#item').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).show();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(' Class:  ' + data["class"]);
                $('#score').fadeIn(600);
                $('#score').text(' Score:  ' + data["score"]);
                $('#filename').fadeIn(600);
                $('#filename').text(' File Name:  ' + data["filename"]);
                $('#time_predict').fadeIn(600);
                $('#time_predict').text(' Time Predict:  ' + data["time_predict"]);
                $('#status').fadeIn(600);
                $('#status').text(' Status:  ' + data["status"]);
                $('#item').fadeIn(600);
                var s = data["result"]
                var myJson = JSON.stringify(s)
                $('#item').text(' Item:  ' + myJson);
                console.log('Success!');
            },
        });
    });

});
