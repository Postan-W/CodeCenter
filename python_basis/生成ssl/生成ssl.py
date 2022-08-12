from OpenSSL import crypto, SSL

#common_name可以填公网ip、内网ip
"""
自建的ssl在Linux用curl测试时要加上-k参数
"""
#在fastapi中使用例如：uvicorn.run(app, host="0.0.0.0", port=10091,debug=True,ssl_keyfile="./key.pem", ssl_certfile="./cert.pem")
#在flask中使用例如：app.run(host="0.0.0.0",port=5001,ssl_context=("./babycuri.com_bundle.crt","./babycuri.com.key"),debug=True)
def generate_certificate(
		organization="PrivacyFilter",
		common_name="10.69.34.22",
		country="NL",
		duration=(365 * 24 * 60 * 60),
		keyfilename="key.pem",
		certfilename="cert.pem"):
	k = crypto.PKey()
	k.generate_key(crypto.TYPE_RSA, 4096)

	cert = crypto.X509()
	cert.get_subject().C = country
	cert.get_subject().O = organization
	cert.get_subject().CN = common_name
	cert.gmtime_adj_notBefore(0)
	cert.gmtime_adj_notAfter(duration)
	cert.set_issuer(cert.get_subject())
	cert.set_pubkey(k)
	cert.sign(k, 'sha512')

	with open(keyfilename, "wt") as keyfile:
		keyfile.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k).decode("utf-8"))
	with open(certfilename, "wt") as certfile:
		certfile.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert).decode("utf-8"))


if __name__ == '__main__':
	generate_certificate()


